"""
Candidate 1: EGT-Net (Evidential Graph-Temporal Network).

Per-bus GRU temporal encoder -> 2-layer GCN over the grid adjacency graph ->
  (a) global-mean-pooled Dirichlet evidential detection head (single forward
      pass, closed-form epistemic uncertainty, no ensembling/MC-sampling), and
  (b) per-bus sigmoid multi-label localization head, jointly trained against
      ground-truth attacked-bus sets.

See MODEL_CANDIDATES.md for the full design rationale and ablation plan.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_features import build_bus_features, normalized_adjacency, incidence_mean_matrix, GCNLayer
from models.evidential import EvidentialHead, dirichlet_predict, evidential_loss


class EGTNet(nn.Module):
    def __init__(self, n_branch: int, n_bus: int, branches, f_bus: int = 6,
                 gru_hidden: int = 32, gcn_hidden: int = 32, use_gcn: bool = True,
                 use_evidential: bool = True):
        super().__init__()
        self.n_branch, self.n_bus = n_branch, n_bus
        self.branches = branches
        inc_mean_np, degree_np = incidence_mean_matrix(n_bus, n_branch, branches)
        self.register_buffer("inc_mean", torch.tensor(inc_mean_np))
        self.register_buffer("degree", torch.tensor(degree_np, dtype=torch.float32))
        self.use_gcn = use_gcn
        self.use_evidential = use_evidential

        self.gru = nn.GRU(f_bus, gru_hidden, batch_first=True)
        A_norm = torch.tensor(normalized_adjacency(n_bus, branches), dtype=torch.float32)
        if use_gcn:
            self.gcn1 = GCNLayer(gru_hidden, gcn_hidden, A_norm)
            self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden, A_norm)
            node_dim = gcn_hidden
        else:
            self.proj = nn.Linear(gru_hidden, gcn_hidden)
            node_dim = gcn_hidden

        if use_evidential:
            self.det_head = EvidentialHead(node_dim, 2)
        else:
            self.det_head = nn.Linear(node_dim, 2)
        self.loc_head = nn.Linear(node_dim, 1)

    def node_embeddings(self, Z: torch.Tensor) -> torch.Tensor:
        feat = build_bus_features(Z, self.n_branch, self.n_bus, self.inc_mean, self.degree)  # (B,k,n_bus,F)
        B, k, n_bus, Fb = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B * n_bus, k, Fb)
        _, h = self.gru(feat)
        h = h[-1].view(B, n_bus, -1)  # (B, n_bus, gru_hidden)
        if self.use_gcn:
            h = F.relu(self.gcn1(h))
            h = F.relu(self.gcn2(h))
        else:
            h = F.relu(self.proj(h))
        return h  # (B, n_bus, node_dim)

    def forward(self, Z: torch.Tensor):
        h = self.node_embeddings(Z)
        pooled = h.mean(dim=1)
        det_out = self.det_head(pooled)
        loc_logits = self.loc_head(h).squeeze(-1)  # (B, n_bus)
        return det_out, loc_logits

    def detect(self, Z: torch.Tensor):
        det_out, _ = self.forward(Z)
        if self.use_evidential:
            p, u = dirichlet_predict(det_out)
            return p[:, 1], u  # P(attacked), epistemic uncertainty
        probs = F.softmax(det_out, dim=-1)
        return probs[:, 1], 1.0 - probs.max(-1).values


def focal_bce(logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """Focal loss for extreme per-bus class imbalance (most buses unattacked)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.where(y == 1, p, 1 - p)
    alpha_t = torch.where(y == 1, alpha, 1 - alpha)
    loss = alpha_t * (1 - pt).pow(gamma) * ce
    return loss.mean()


def egt_loss(model: EGTNet, Z, y, ybus, epoch, total_epochs, gamma_loc=1.0, kl_max=1.0):
    det_out, loc_logits = model(Z)
    if model.use_evidential:
        l_det = evidential_loss(det_out, y, epoch, total_epochs, kl_max=kl_max)
    else:
        l_det = F.cross_entropy(det_out, y)
    l_loc = focal_bce(loc_logits, ybus)
    return l_det + gamma_loc * l_loc, dict(l_det=l_det.item(), l_loc=l_loc.item())
