"""
Dirichlet evidential-deep-learning classification head (Sensoy, Kandemir &
Kaplan, NeurIPS 2018): a single forward pass produces non-negative
"evidence" e_c per class, Dirichlet parameters alpha_c = e_c + 1, predictive
p_c = alpha_c / S (S = sum alpha), and a closed-form epistemic-uncertainty
score u = C / S -- no ensembling, no MC sampling, no Laplace fitting. This is
Candidate 1/2's (EGT-Net) detection-uncertainty mechanism, replacing the
source paper's multiheaded-ensemble + LLLA + MC-integration pipeline with a
single-forward-pass alternative (see SOURCE_WEAKNESS_ANALYSIS.md, Rank 4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        evidence = F.softplus(self.fc(x))
        alpha = evidence + 1.0
        return alpha


def dirichlet_predict(alpha: torch.Tensor):
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    u = alpha.shape[-1] / S.squeeze(-1)  # epistemic uncertainty, Sensoy et al. Eq. 7
    return p, u


def evidential_loss(alpha: torch.Tensor, y: torch.Tensor, epoch: int, total_epochs: int,
                     kl_max: float = 1.0) -> torch.Tensor:
    """Sum-of-squares Bayes-risk loss + annealed KL(Dir(alpha~)||Dir(1)) term
    (Sensoy et al. 2018, Eqs. 5-6)."""
    n_classes = alpha.shape[-1]
    y_onehot = F.one_hot(y, n_classes).float()
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    err = (y_onehot - p).pow(2).sum(-1)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(-1)
    loss_sq = err + var

    lam_t = min(kl_max, kl_max * epoch / max(1, total_epochs))
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha  # remove evidence on the *correct* class
    kl = _kl_dirichlet_uniform(alpha_tilde)
    return (loss_sq + lam_t * kl).mean()


def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    n_classes = alpha.shape[-1]
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(-1, keepdim=True)
    S_ones = torch.tensor(float(n_classes), device=alpha.device)
    term1 = torch.lgamma(S_alpha).squeeze(-1) - torch.lgamma(alpha).sum(-1) \
        - (torch.lgamma(S_ones) - torch.lgamma(ones).sum(-1))
    term2 = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(-1)
    return term1 + term2
