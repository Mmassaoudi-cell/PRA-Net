"""
Shared per-bus featurization and a lightweight (dependency-free) GCN layer,
used by the GCN benchmark and by all EGT-Net-family candidates (see
MODEL_CANDIDATES.md). Maps the flat (k,d) measurement-channel representation
onto a fixed-width per-bus feature vector so that graph/message-passing
layers of consistent width can be used regardless of system size (30 vs 118
buses).

Per-bus feature vector (F_bus=6): [P_inj, Q_inj, Vmag, mean(P_flow of
incident branches), mean(Q_flow of incident branches), degree/10].
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def normalized_adjacency(n_bus: int, branches: list[tuple[int, int, float]]) -> np.ndarray:
    A = np.eye(n_bus)
    for fb, tb, _ in branches:
        A[fb, tb] = 1.0
        A[tb, fb] = 1.0
    deg = A.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def incidence_lists(n_bus: int, branches: list[tuple[int, int, float]]):
    inc = [[] for _ in range(n_bus)]
    for idx, (fb, tb, _) in enumerate(branches):
        inc[fb].append(idx)
        inc[tb].append(idx)
    return inc


def incidence_mean_matrix(n_bus: int, n_branch: int, branches: list[tuple[int, int, float]]) -> np.ndarray:
    """(n_bus, n_branch) matrix M with M[b, br] = 1/degree(b) if branch br is
    incident to bus b, else 0 -- so `p_flow @ M.T` computes, for every bus in
    one matmul, the mean of its incident branches' flows (no Python loop)."""
    M = np.zeros((n_bus, n_branch), dtype=np.float32)
    for br, (fb, tb, _) in enumerate(branches):
        M[fb, br] += 1.0
        M[tb, br] += 1.0
    deg = M.sum(axis=1, keepdims=True)
    deg_safe = np.where(deg > 0, deg, 1.0)
    return M / deg_safe, deg.squeeze(-1)


def build_bus_features(Z: torch.Tensor, n_branch: int, n_bus: int,
                        inc_mean: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    """Fully vectorized (no per-bus Python loop). Z: (B, k, d) -> (B, k, n_bus, 6).
    `inc_mean` (n_bus, n_branch) and `degree` (n_bus,) come from
    `incidence_mean_matrix`, precomputed once per network and passed in as
    registered buffers by the caller (see candidates/egt_net.py etc.)."""
    p_flow = Z[..., :n_branch]
    q_flow = Z[..., n_branch:2 * n_branch]
    base = 2 * n_branch
    p_inj = Z[..., base:base + n_bus]
    q_inj = Z[..., base + n_bus:base + 2 * n_bus]
    vmag = Z[..., base + 2 * n_bus:base + 3 * n_bus]

    p_flow_mean = p_flow @ inc_mean.T   # (B,k,n_branch) @ (n_branch,n_bus) -> (B,k,n_bus)
    q_flow_mean = q_flow @ inc_mean.T
    B, k = Z.shape[0], Z.shape[1]
    degree_feat = (degree / 10.0).view(1, 1, n_bus).expand(B, k, n_bus)
    feat = torch.stack([p_inj, q_inj, vmag, p_flow_mean, q_flow_mean, degree_feat], dim=-1)
    return feat  # (B, k, n_bus, 6)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, A_norm: torch.Tensor):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.register_buffer("A_norm", A_norm)

    def forward(self, x):
        # x: (B, n_bus, in_dim)
        h = self.lin(x)
        return torch.einsum("ij,bjf->bif", self.A_norm, h)
