"""
Reproduction of the source paper's core proposed method (Sections III-IV):
multiheaded ensemble network trained with the ensemble-aware CE + JSD
diversity + ensemble-entropy loss (Eqs. 6-9), extended with a manual
last-layer Laplace approximation (LLLA, Eq. 11-12) computed via the exact
generalized-Gauss-Newton (GGN) Hessian on each head's final linear layer,
and epistemic-uncertainty (mutual information) scoring (Eq. 10).

We implement LLLA manually (rather than via the `laplace-torch` package)
because that package's `curvlinops` dependency is incompatible with this
project's torch 2.12 dev build; the exact-GGN last-layer closed form below
is the same object the package would compute (Kristiadi et al. 2020,
the source paper's own cited reference [28]), just without the library.

Also implements the two comparison UQ baselines used in the source paper's
own Table IV: MC dropout (T stochastic forward passes) and Deep Ensembles
(m independently trained models).
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multiheaded ensemble + training losses (Eqs. 6-9)
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.last = nn.Linear(hidden, 2)  # targeted by LLLA

    def penultimate(self, feat):
        return self.body(feat)

    def forward(self, feat):
        return self.last(self.body(feat))


class MultiHeadEnsemble(nn.Module):
    def __init__(self, backbone: nn.Module, m: int = 5, hidden: int = 64):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.heads = nn.ModuleList([Head(backbone.out_dim, hidden) for _ in range(m)])
        self.m = m

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone.features(x)
        return [h(feat) for h in self.heads]  # list of (B,2) logits


def jsd_diversity_loss(probs_list):
    """Eq. 7: -mean pairwise Jensen-Shannon divergence between heads' softmax."""
    m = len(probs_list)
    total, cnt = 0.0, 0
    for i in range(m):
        for j in range(i + 1, m):
            pij = 0.5 * (probs_list[i] + probs_list[j])
            kl_i = F.kl_div(pij.clamp_min(1e-8).log(), probs_list[i], reduction="none").sum(-1)
            kl_j = F.kl_div(pij.clamp_min(1e-8).log(), probs_list[j], reduction="none").sum(-1)
            jsd = 0.5 * (kl_i + kl_j)
            total = total + jsd.mean()
            cnt += 1
    return -(total / max(cnt, 1))


def multihead_loss(logits_list, y, lam_ece=0.1, lam_div=0.9, lam_ee=1.0):
    """Eq. 9: total = lam_ece*L_ece + lam_div*L_div + lam_ee*L_ee."""
    probs_list = [F.softmax(l, dim=-1) for l in logits_list]
    m = len(logits_list)
    avg_logit_probs = torch.stack(probs_list, 0).mean(0).clamp_min(1e-8)
    ece = F.nll_loss(avg_logit_probs.log(), y) + sum(F.cross_entropy(l, y) for l in logits_list) / m
    div = jsd_diversity_loss(probs_list)
    # Eq. 8: L_ee = P1*log(P1) + P0*log(P0) = -H(P); minimizing L_ee therefore
    # maximizes the entropy (smoothness) of the ensemble-averaged softmax.
    ee = (avg_logit_probs * avg_logit_probs.log()).sum(-1).mean()
    total = lam_ece * ece + lam_div * div + lam_ee * ee
    return total, dict(ece=ece.item(), div=div.item(), ee=ee.item())


def mutual_information(probs_stack: torch.Tensor) -> torch.Tensor:
    """Eq. 10: I(Y;Theta|Z) = H(mean_i p_i) - mean_i H(p_i).
    probs_stack: (m, B, C)."""
    p_bar = probs_stack.mean(0).clamp_min(1e-12)
    h_mean = -(p_bar * p_bar.log()).sum(-1)
    h_each = -(probs_stack.clamp_min(1e-12) * probs_stack.clamp_min(1e-12).log()).sum(-1).mean(0)
    return h_mean - h_each


# ---------------------------------------------------------------------------
# Manual last-layer Laplace approximation (exact GGN, closed form)
# ---------------------------------------------------------------------------

class LastLayerLaplace:
    """Exact-GGN Laplace approximation over one linear layer's weights+bias
    (Kristiadi et al. 2020 'last-layer Laplace'), computed in closed form
    since the layer is small (feat->2)."""

    def __init__(self, prior_precision: float = 1.0):
        self.prior_precision = prior_precision
        self.mean = None   # (C, F+1) flattened -> (C*(F+1),)
        self.cov = None    # (C*(F+1), C*(F+1))
        self.C = 2

    @torch.no_grad()
    def fit(self, penult_feats: torch.Tensor, logits: torch.Tensor, last_layer: nn.Linear):
        """penult_feats: (N, F) penultimate features (input to `last_layer`).
        logits: (N, C) = last_layer(penult_feats)."""
        device = penult_feats.device
        # float64 for numerical stability: the GGN Hessian accumulates outer
        # products over the full training set, which can be ill-conditioned
        # in float32 for wide feature ranges / near-saturated softmax probs.
        penult_feats = penult_feats.double()
        logits = logits.double()
        N, F_ = penult_feats.shape
        C = logits.shape[1]
        self.C = C
        ones = torch.ones(N, 1, device=device, dtype=torch.float64)
        a_ext = torch.cat([penult_feats, ones], dim=1)  # (N, F+1)
        Fp1 = F_ + 1
        p = F.softmax(logits, dim=-1)  # (N, C)

        # Block Hessian: H[c,c'] = sum_i Lambda_i[c,c'] * a_i a_i^T (Fp1 x Fp1
        # each), accumulated in chunks to bound peak memory for large N.
        H = torch.zeros(C * Fp1, C * Fp1, device=device, dtype=torch.float64)
        chunk = 2000
        for s in range(0, N, chunk):
            a_c = a_ext[s:s + chunk]
            p_c = p[s:s + chunk]
            outer = torch.einsum("ni,nj->nij", a_c, a_c)  # (n, Fp1, Fp1)
            for c in range(C):
                for cp in range(C):
                    lam = p_c[:, c] * ((1.0 if c == cp else 0.0) - p_c[:, cp])
                    H[c * Fp1:(c + 1) * Fp1, cp * Fp1:(cp + 1) * Fp1] += torch.einsum("n,nij->ij", lam, outer)

        eye = torch.eye(C * Fp1, device=device, dtype=torch.float64)
        prec = self.prior_precision
        cov = None
        for _ in range(6):
            Hp = H + prec * eye
            if torch.isfinite(Hp).all():
                try:
                    cov = torch.linalg.inv(Hp)
                    if torch.isfinite(cov).all():
                        break
                except RuntimeError:
                    pass
            prec *= 10.0  # singular/ill-conditioned: strengthen the prior and retry
        if cov is None:
            cov = eye / prec
        self.cov = cov.float()
        W = last_layer.weight.data  # (C, F)
        b = last_layer.bias.data    # (C,)
        mean = torch.cat([W, b.unsqueeze(1)], dim=1).reshape(-1)  # (C*Fp1,)
        self.mean = mean.float()
        self.Fp1 = Fp1

    @torch.no_grad()
    def predictive_gaussian(self, penult_feats: torch.Tensor):
        """Closed-form Gaussian-process-style predictive (no MC sampling):
        for logit c, mean_c = w_c . a_ext, var_c = a_ext^T Sigma_cc a_ext.
        Used by the SNGP-style candidate (EGTNetSNGP) as its uncertainty
        mechanism in place of Monte-Carlo integration of Eq. 12."""
        device = penult_feats.device
        N = penult_feats.shape[0]
        ones = torch.ones(N, 1, device=device)
        a_ext = torch.cat([penult_feats, ones], dim=1)
        C, Fp1 = self.C, self.Fp1
        W = self.mean.view(C, Fp1)
        mean_logits = a_ext @ W.T  # (N, C)
        var_logits = torch.zeros(N, C, device=device)
        for c in range(C):
            Sigma_cc = self.cov[c * Fp1:(c + 1) * Fp1, c * Fp1:(c + 1) * Fp1]
            var_logits[:, c] = ((a_ext @ Sigma_cc) * a_ext).sum(-1)
        return mean_logits, var_logits

    @torch.no_grad()
    def predictive(self, penult_feats: torch.Tensor, mc_samples: int = 30) -> torch.Tensor:
        """Eq. 12 via Monte-Carlo integration: E_w~N(mean,cov)[softmax(w.a)]."""
        return self.predictive_with_grad(penult_feats, mc_samples)

    def predictive_with_grad(self, penult_feats: torch.Tensor, mc_samples: int = 30) -> torch.Tensor:
        """Same as `predictive`, but without a no_grad guard, so gradients can
        flow back through `penult_feats` to the original input Z -- needed by
        the localization stage's input-gradient attribution (Eq. 16). The
        sampled weights themselves are treated as constants (no grad needed
        there); only the feature path must stay differentiable."""
        device = penult_feats.device
        N = penult_feats.shape[0]
        ones = torch.ones(N, 1, device=device)
        a_ext = torch.cat([penult_feats, ones], dim=1)
        C, Fp1 = self.C, self.Fp1
        with torch.no_grad():
            L = torch.linalg.cholesky(self.cov + 1e-6 * torch.eye(self.cov.shape[0], device=device))
            eps = torch.randn(mc_samples, self.mean.shape[0], device=device)
            samples = self.mean.unsqueeze(0) + eps @ L.T  # (S, C*Fp1)
            W = samples.view(mc_samples, C, Fp1)
        logits = torch.einsum("scf,nf->snc", W, a_ext)  # (S, N, C)
        probs = F.softmax(logits, dim=-1).mean(0)  # (N, C)
        return probs


class LLLAMultiHeadEnsemble:
    """Wraps a trained MultiHeadEnsemble with per-head LastLayerLaplace."""

    def __init__(self, ensemble: MultiHeadEnsemble, prior_precision: float = 1.0):
        self.ensemble = ensemble
        self.llla = [LastLayerLaplace(prior_precision) for _ in range(ensemble.m)]

    @torch.no_grad()
    def fit(self, loader, device):
        self.ensemble.eval()
        feats_all, logits_all = [[] for _ in range(self.ensemble.m)], [[] for _ in range(self.ensemble.m)]
        for xb, yb in loader:
            xb = xb.to(device)
            feat = self.ensemble.backbone.features(xb)
            for i, head in enumerate(self.ensemble.heads):
                pen = head.penultimate(feat)
                feats_all[i].append(pen)
                logits_all[i].append(head.last(pen))
        for i, head in enumerate(self.ensemble.heads):
            pen = torch.cat(feats_all[i], 0)
            log = torch.cat(logits_all[i], 0)
            self.llla[i].fit(pen, log, head.last)

    @torch.no_grad()
    def head_predictives(self, x, mc_samples: int = 30):
        return self.head_predictives_with_grad(x, mc_samples)

    def head_predictives_with_grad(self, x, mc_samples: int = 30):
        """Backbone is frozen (requires_grad=False on its params) but is
        called *without* torch.no_grad() here so the autograd graph from `x`
        through backbone.features -> head.penultimate -> LLLA predictive
        stays intact -- required for Eq. 16's d(MI)/d(Z) attribution."""
        feat = self.ensemble.backbone.features(x)
        probs = []
        for i, head in enumerate(self.ensemble.heads):
            pen = head.penultimate(feat)
            probs.append(self.llla[i].predictive_with_grad(pen, mc_samples))
        return torch.stack(probs, 0)  # (m, B, C)

    @torch.no_grad()
    def mi(self, x, mc_samples: int = 30):
        return mutual_information(self.head_predictives(x, mc_samples))

    def mi_with_grad(self, x, mc_samples: int = 30):
        return mutual_information(self.head_predictives_with_grad(x, mc_samples))


# ---------------------------------------------------------------------------
# Baseline 1: MC dropout
# ---------------------------------------------------------------------------

def enable_mc_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_mi(model: nn.Module, x: torch.Tensor, T: int = 50) -> torch.Tensor:
    model.eval()
    enable_mc_dropout(model)
    probs = torch.stack([F.softmax(model(x), dim=-1) for _ in range(T)], 0)
    return mutual_information(probs)


# ---------------------------------------------------------------------------
# Baseline 2: Deep ensembles (m independently trained models)
# ---------------------------------------------------------------------------

@torch.no_grad()
def deep_ensemble_mi(models: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    probs = torch.stack([F.softmax(m(x), dim=-1) for m in models], 0)
    return mutual_information(probs)
