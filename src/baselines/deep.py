"""
Deep-learning and graph benchmark suite beyond the source paper's own five
backbones (which already serve as LSTM/GRU/CNN/ResNet/Transformer
supervised baselines via their `test_clean_metrics` in the reproduction
JSON): TCN (temporal, dilated-causal) and a plain GCN (task-specific,
topology-aware) classifier. Share the (B,k,d)->logits(B,2) interface used by
train/common.py so they plug into the same train/eval loop as the backbones.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_features import build_bus_features, normalized_adjacency, incidence_mean_matrix, GCNLayer


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.net = nn.Sequential(self.conv1, self.chomp1, nn.ReLU(), nn.Dropout(dropout),
                                  self.conv2, self.chomp2, nn.ReLU(), nn.Dropout(dropout))
        self.short = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.short(x))


class TCNBackbone(nn.Module):
    """Causal dilated-convolution temporal backbone (Bai et al. 2018)."""
    def __init__(self, d: int, channels=(64, 64, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        blocks, c_in = [], d
        for i, c_out in enumerate(channels):
            blocks.append(TCNBlock(c_in, c_out, kernel_size, dilation=2 ** i, dropout=dropout))
            c_in = c_out
        self.net = nn.Sequential(*blocks)
        self.out_dim = c_in
        self.fc = nn.Linear(c_in, 2)

    def features(self, x):
        h = self.net(x.transpose(1, 2))
        return h[:, :, -1]  # causal: last timestep summarizes the whole window

    def forward(self, x):
        return self.fc(self.features(x))


class GCNBenchmark(nn.Module):
    """Plain (non-evidential, detection-only) 2-layer GCN classifier -- the
    task-specific graph baseline for the main benchmark table."""
    def __init__(self, n_branch, n_bus, branches, f_bus=6, hidden=32):
        super().__init__()
        self.n_branch, self.n_bus, self.branches = n_branch, n_bus, branches
        inc_mean_np, degree_np = incidence_mean_matrix(n_bus, n_branch, branches)
        self.register_buffer("inc_mean", torch.tensor(inc_mean_np))
        self.register_buffer("degree", torch.tensor(degree_np, dtype=torch.float32))
        A_norm = torch.tensor(normalized_adjacency(n_bus, branches), dtype=torch.float32)
        self.proj = nn.Linear(f_bus * 6, hidden)  # flatten k=6 timesteps per bus
        self.gcn1 = GCNLayer(hidden, hidden, A_norm)
        self.gcn2 = GCNLayer(hidden, hidden, A_norm)
        self.out_dim = hidden
        self.fc = nn.Linear(hidden, 2)

    def features(self, x):
        feat = build_bus_features(x, self.n_branch, self.n_bus, self.inc_mean, self.degree)  # (B,k,n_bus,6)
        B, k, n_bus, Fb = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B, n_bus, k * Fb)
        h = F.relu(self.proj(feat))
        h = F.relu(self.gcn1(h))
        h = F.relu(self.gcn2(h))
        return h.mean(dim=1)

    def forward(self, x):
        return self.fc(self.features(x))
