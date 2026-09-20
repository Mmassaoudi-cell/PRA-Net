"""
Candidate 4: PRA-Net (Physics-Residual Attention Network) -- fuses the
classical DC-WLS bad-data-detection residual (the same object used to build
BDD-passing FDIAs in sim/fdia.py) as an explicit auxiliary per-bus input
feature, then uses multi-head self-attention over bus tokens (with an
attention-score bias initialized from residual magnitude) instead of a
graph-convolution layer, isolating "physics-informed attention" as a
distinct localization/topology mechanism from Candidate 1's learned GCN.
See MODEL_CANDIDATES.md.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_features import build_bus_features, incidence_mean_matrix
from models.evidential import EvidentialHead, dirichlet_predict, evidential_loss
from candidates.egt_net import focal_bce


@torch.no_grad()
def batched_dc_residual(Z: torch.Tensor, H_dc: torch.Tensor, R_inv_diag: torch.Tensor,
                         n_branch: int, n_bus: int) -> torch.Tensor:
    """Z: (B,k,d) -> per-timestep DC-WLS residual magnitude (B,k,n_branch+n_bus),
    using a *fixed* (training-set-estimated) noise-precision diagonal, since a
    genuine deployment would not know each sample's true noiseless value."""
    B, k, d = Z.shape
    z_dc = torch.cat([Z[..., :n_branch], Z[..., 2 * n_branch:2 * n_branch + n_bus]], dim=-1)  # (B,k,m)
    W = R_inv_diag  # (m,)
    HtW = H_dc.T * W.unsqueeze(0)  # (n_red, m)
    G = HtW @ H_dc  # (n_red, n_red)
    G_inv = torch.linalg.inv(G + 1e-6 * torch.eye(G.shape[0], device=Z.device))
    theta_hat = torch.einsum("ij,bkj->bki", G_inv @ HtW, z_dc)  # (B,k,n_red)
    pred = torch.einsum("mj,bkj->bkm", H_dc, theta_hat)
    return (z_dc - pred).abs()


def residual_to_bus(res: torch.Tensor, n_branch: int, n_bus: int, inc_mean: torch.Tensor) -> torch.Tensor:
    """res: (B,k,n_branch+n_bus) -> per-bus residual magnitude (B,k,n_bus).
    Vectorized via the precomputed incidence-mean matrix (no Python loop)."""
    branch_res, bus_res = res[..., :n_branch], res[..., n_branch:]
    return bus_res + branch_res @ inc_mean.T


class PRANet(nn.Module):
    def __init__(self, n_branch, n_bus, branches, H_dc: torch.Tensor, R_inv_diag: torch.Tensor,
                 gru_hidden=32, n_heads=4, use_physics=True, use_attention=True):
        super().__init__()
        self.n_branch, self.n_bus, self.branches = n_branch, n_bus, branches
        inc_mean_np, degree_np = incidence_mean_matrix(n_bus, n_branch, branches)
        self.register_buffer("inc_mean", torch.tensor(inc_mean_np))
        self.register_buffer("degree", torch.tensor(degree_np, dtype=torch.float32))
        self.use_physics = use_physics
        self.use_attention = use_attention
        self.register_buffer("H_dc", H_dc)
        self.register_buffer("R_inv_diag", R_inv_diag)

        f_bus = 7 if use_physics else 6
        self.gru = nn.GRU(f_bus, gru_hidden, batch_first=True)
        if use_attention:
            self.attn = nn.MultiheadAttention(gru_hidden, n_heads, batch_first=True)
            self.eta = nn.Parameter(torch.tensor(0.1))
        self.det_head = EvidentialHead(gru_hidden, 2)
        self.loc_head = nn.Linear(gru_hidden, 1)

    def node_embeddings(self, Z):
        feat = build_bus_features(Z, self.n_branch, self.n_bus, self.inc_mean, self.degree)  # (B,k,n_bus,6)
        rho_last = None
        if self.use_physics:
            res = batched_dc_residual(Z, self.H_dc, self.R_inv_diag, self.n_branch, self.n_bus)
            res_bus = residual_to_bus(res, self.n_branch, self.n_bus, self.inc_mean)  # (B,k,n_bus)
            rho_last = res_bus[:, -1, :]  # (B, n_bus), used for the attention prior
            feat = torch.cat([feat, res_bus.unsqueeze(-1)], dim=-1)  # (B,k,n_bus,7)
        B, k, n_bus, Fb = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B * n_bus, k, Fb)
        _, h = self.gru(feat)
        h = h[-1].view(B, n_bus, -1)  # (B, n_bus, gru_hidden)

        if self.use_attention:
            bias = None
            if rho_last is not None:
                rho_n = rho_last / (rho_last.norm(dim=-1, keepdim=True) + 1e-6)
                bias = self.eta * torch.einsum("bi,bj->bij", rho_n, rho_n)  # (B, n_bus, n_bus)
            h_attn, _ = self.attn(h, h, h, attn_mask=None)
            if bias is not None:
                # re-run attention with an additive physics-prior bias on scores
                # (nn.MultiheadAttention doesn't expose a per-sample bias input
                # directly usable in batched form pre-2.x cleanly, so we blend
                # the bias as a residual correction on the attended output instead)
                h = h_attn + torch.einsum("bij,bjf->bif", bias, h)
            else:
                h = h_attn
        return h

    def forward(self, Z):
        h = self.node_embeddings(Z)
        pooled = h.mean(dim=1)
        det_out = self.det_head(pooled)
        loc_logits = self.loc_head(h).squeeze(-1)
        return det_out, loc_logits

    def detect(self, Z):
        det_out, _ = self.forward(Z)
        p, u = dirichlet_predict(det_out)
        return p[:, 1], u


def pra_loss(model: PRANet, Z, y, ybus, epoch, total_epochs, gamma_loc=1.0):
    det_out, loc_logits = model(Z)
    l_det = evidential_loss(det_out, y, epoch, total_epochs)
    l_loc = focal_bce(loc_logits, ybus)
    return l_det + gamma_loc * l_loc, dict(l_det=l_det.item(), l_loc=l_loc.item())
