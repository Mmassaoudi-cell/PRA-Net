"""
Reproduction of the source paper's uncertainty-guided localization method
(Sec. IV-B, Eqs. 14-18): a VAE learns the manifold of measurement sequences;
given a detected adversarial sequence Z^adv, we search in VAE latent space
for a nearby, lower-epistemic-uncertainty counterfactual Z^-, then attribute
the resulting uncertainty reduction to measurement channels via gradients of
the mutual-information (MI) function, aggregate to buses, and apply a
topology-based neighborhood-consistency filter (only buses whose |rank
difference| to *all* graph neighbors is exactly 1 keep their score).

Implementation note on the L0 term in Eq. 14: the true L0 "norm" has zero
gradient almost everywhere, so it cannot directly drive gradient-based
optimization. We use the standard smooth surrogate
    L0_smooth(v) = sum_i v_i^2 / (v_i^2 + eps)
which saturates to 1 for any non-negligible |v_i| and to 0 as v_i -> 0,
approximating a differentiable sparsity count -- documented as an
implementation assumption (SOURCE_PAPER_AUDIT.md, "Partially Specified").
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqVAE(nn.Module):
    def __init__(self, k: int, d: int, latent_dim: int = 32, hidden=(512, 128)):
        super().__init__()
        in_dim = k * d
        self.k, self.d = k, d
        h1, h2 = hidden
        self.enc = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, in_dim))

    def encode(self, x):
        h = self.enc(x.reshape(x.shape[0], -1))
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        out = self.dec(z)
        return out.reshape(-1, self.k, self.d)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar


def vae_loss(x, x_hat, mu, logvar):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 1e-3 * kl, recon.item(), kl.item()


def counterfactual_search(vae: SeqVAE, llla_ensemble, Z_adv: torch.Tensor,
                           beta1=2.0, beta2=2.0, beta3=1.0, lr=2e-2, steps=5, mc_samples=20):
    """Eq. 14. Z_adv: (B,k,d). Returns Z_minus (B,k,d)."""
    with torch.no_grad():
        mu_adv, _ = vae.encode(Z_adv)
    gamma = mu_adv.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([gamma], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        Z_minus = vae.decode(gamma)
        mi = llla_ensemble.mi_with_grad(Z_minus, mc_samples=mc_samples).mean()
        l1 = (gamma - mu_adv).abs().sum(-1).mean()
        dz = Z_minus - Z_adv
        l0_smooth = (dz.pow(2) / (dz.pow(2) + 1e-6)).sum(dim=(1, 2)).mean()
        loss = beta1 * mi + beta2 * l1 + beta3 * l0_smooth
        loss.backward()
        opt.step()
    with torch.no_grad():
        Z_minus = vae.decode(gamma)
    return Z_minus.detach()


def channel_contribution_gradients(llla_ensemble, Z_adv: torch.Tensor, mc_samples=20):
    """Eq. 16: G_{k,n} = d I(Y;Theta|Z) / d z_{k,n}, evaluated at Z_adv."""
    Z = Z_adv.clone().detach().requires_grad_(True)
    mi = llla_ensemble.mi_with_grad(Z, mc_samples=mc_samples).sum()
    grad, = torch.autograd.grad(mi, Z)
    return grad  # (B, k, d)


def localization_scores(G: torch.Tensor, dZ: torch.Tensor, n_branch: int, n_bus: int,
                         branches: list[tuple[int, int, float]]) -> np.ndarray:
    """Eqs. 15-18: contribution -> channel score -> bus score. Returns (B, n_bus)."""
    contrib = (G.abs() * dZ.abs()).sum(dim=1)  # sum over k -> (B, d) = s_n (Eq. 17)
    B, d = contrib.shape
    bus_scores = torch.zeros(B, n_bus, device=contrib.device)
    for br_idx, (fb, tb, _) in enumerate(branches):
        bus_scores[:, fb] += contrib[:, br_idx]                      # P_flow
        bus_scores[:, tb] += contrib[:, br_idx]
        bus_scores[:, fb] += contrib[:, n_branch + br_idx]            # Q_flow
        bus_scores[:, tb] += contrib[:, n_branch + br_idx]
    base = 2 * n_branch
    bus_scores += contrib[:, base: base + n_bus]                      # P_inj
    bus_scores += contrib[:, base + n_bus: base + 2 * n_bus]          # Q_inj
    bus_scores += contrib[:, base + 2 * n_bus: base + 3 * n_bus]      # Vmag
    return bus_scores.cpu().numpy()


def neighborhood_consistency_filter(bus_scores: np.ndarray, graph) -> np.ndarray:
    """Zero out a bus's score unless min |rank diff| to any graph neighbor == 1."""
    n_bus = bus_scores.shape[0]
    order = np.argsort(-bus_scores)
    rank = np.empty(n_bus, dtype=int)
    rank[order] = np.arange(n_bus)
    out = bus_scores.copy()
    for b in range(n_bus):
        neighbors = list(graph.neighbors(b))
        if not neighbors:
            out[b] = 0.0
            continue
        diffs = [abs(rank[b] - rank[nb]) for nb in neighbors]
        if min(diffs) != 1:
            out[b] = 0.0
    return out


def calibrate_localization_threshold(bus_scores_matrix: np.ndarray, top_frac=0.30, alpha=0.95) -> float:
    """Table VI recipe: for each historical non-attacked sequence, average its
    top-`top_frac` bus scores, then take the `alpha` percentile across sequences."""
    n_seq, n_bus = bus_scores_matrix.shape
    k = max(1, int(n_bus * top_frac))
    top_means = np.sort(bus_scores_matrix, axis=1)[:, -k:].mean(axis=1)
    return float(np.percentile(top_means, alpha * 100))
