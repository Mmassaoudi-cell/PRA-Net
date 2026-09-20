"""
Candidate 2: EGT-Net/SNGP -- same GRU+GCN backbone as Candidate 1 (egt_net.py)
but the detection head is a spectral-normalized random-Fourier-feature (RFF)
approximation of a Gaussian process (Liu et al. 2020, "Simple and Principled
Uncertainty Estimation with Deterministic Deep Learning via Distance
Awareness"), with a closed-form last-layer-Laplace posterior (reusing
models/uncertainty.LastLayerLaplace) giving distance-aware uncertainty
without any Monte-Carlo sampling. See MODEL_CANDIDATES.md.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from models.graph_features import build_bus_features, normalized_adjacency, incidence_mean_matrix, GCNLayer
from models.uncertainty import LastLayerLaplace
from candidates.egt_net import focal_bce


class RFFLayer(nn.Module):
    """Random Fourier feature map phi(x) = sqrt(2/D) * cos(W x + b), a
    fixed (non-trained) approximation to an RBF-kernel feature space, on top
    of which a linear "GP" output layer is fit (see SNGP, Liu et al. 2020)."""
    def __init__(self, in_dim: int, n_features: int = 128, length_scale: float = 1.0):
        super().__init__()
        W = torch.randn(in_dim, n_features) / length_scale
        b = torch.rand(n_features) * 2 * math.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.n_features = n_features

    def forward(self, x):
        return math.sqrt(2.0 / self.n_features) * torch.cos(x @ self.W + self.b)


class EGTNetSNGP(nn.Module):
    def __init__(self, n_branch: int, n_bus: int, branches, f_bus: int = 6,
                 gru_hidden: int = 32, gcn_hidden: int = 32, rff_dim: int = 128):
        super().__init__()
        self.n_branch, self.n_bus = n_branch, n_bus
        self.branches = branches
        inc_mean_np, degree_np = incidence_mean_matrix(n_bus, n_branch, branches)
        self.register_buffer("inc_mean", torch.tensor(inc_mean_np))
        self.register_buffer("degree", torch.tensor(degree_np, dtype=torch.float32))

        # Spectral-normalizing the GRU's recurrent weight is deliberately
        # avoided: `parametrize`-based spectral_norm on a recurrent weight
        # defeats cuDNN's fused RNN kernel (forcing a much slower, effectively
        # unfused fallback -- measured 1-2 orders of magnitude slower in this
        # project), and the SNGP method (Liu et al. 2020) itself only
        # requires spectral normalization on the feedforward/residual feature
        # extractor to bound its Lipschitz constant, not on a recurrent core.
        self.gru = nn.GRU(f_bus, gru_hidden, batch_first=True)
        A_norm = torch.tensor(normalized_adjacency(n_bus, branches), dtype=torch.float32)
        self.gcn1 = GCNLayer(gru_hidden, gcn_hidden, A_norm)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden, A_norm)
        for gcn in [self.gcn1, self.gcn2]:
            gcn.lin = spectral_norm(gcn.lin)

        self.rff = RFFLayer(gcn_hidden, rff_dim)
        self.gp_linear = nn.Linear(rff_dim, 2)
        self.loc_head = nn.Linear(gcn_hidden, 1)
        self.llla = None  # fit post-hoc (see fit_gp_laplace)

    def node_embeddings(self, Z):
        feat = build_bus_features(Z, self.n_branch, self.n_bus, self.inc_mean, self.degree)
        B, k, n_bus, Fb = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B * n_bus, k, Fb)
        _, h = self.gru(feat)
        h = h[-1].view(B, n_bus, -1)
        h = F.relu(self.gcn1(h))
        h = F.relu(self.gcn2(h))
        return h

    def forward(self, Z):
        h = self.node_embeddings(Z)
        pooled = h.mean(dim=1)
        phi = self.rff(pooled)
        det_logits = self.gp_linear(phi)
        loc_logits = self.loc_head(h).squeeze(-1)
        return det_logits, loc_logits, phi

    @torch.no_grad()
    def fit_gp_laplace(self, loader, device, prior_precision: float = 5.0):
        self.eval()
        phis, logits = [], []
        for batch in loader:
            xb = batch[0].to(device)
            det_logits, _, phi = self.forward(xb)
            phis.append(phi)
            logits.append(det_logits)
        phis, logits = torch.cat(phis), torch.cat(logits)
        self.llla = LastLayerLaplace(prior_precision)
        self.llla.fit(phis, logits, self.gp_linear)

    @torch.no_grad()
    def detect(self, Z):
        h = self.node_embeddings(Z)
        pooled = h.mean(dim=1)
        phi = self.rff(pooled)
        mean_logits, var_logits = self.llla.predictive_gaussian(phi)
        # mean-field softmax approximation (Lu et al. 2020) folding logit
        # variance into the mean before softmax:
        kappa = 1.0 / torch.sqrt(1.0 + math.pi / 8.0 * var_logits)
        probs = F.softmax(mean_logits * kappa, dim=-1)
        uncertainty = var_logits.mean(-1)
        return probs[:, 1], uncertainty


def sngp_loss(model: EGTNetSNGP, Z, y, ybus, gamma_loc=1.0):
    det_logits, loc_logits, _ = model(Z)
    l_det = F.cross_entropy(det_logits, y)
    l_loc = focal_bce(loc_logits, ybus)
    return l_det + gamma_loc * l_loc, dict(l_det=l_det.item(), l_loc=l_loc.item())
