"""
The five BDD-passing-FDIA sequence-classification backbones from Table I of
the source paper (LSTM, GRU, FCNN [a 1D-CNN despite the name], ResNet, TST).
Each exposes `.features(x)` (the pooled representation before the final FC
layer, used as the frozen backbone for the multiheaded ensemble) and
`.forward(x)` (full logits, used to train/evaluate the plain detector and to
craft the C&W adversarial FDIA).

Input convention: x has shape (batch, k=6, d) exactly as the paper's
Z in R^{k x d}; internally reshaped to (batch, d, k) for the CNN/ResNet.
Hyperparameters follow "commonly adopted settings in time-series
classification" (tsai library defaults, ref [34] of the source paper), since
the paper does not give exact widths -- documented in SOURCE_PAPER_AUDIT.md.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):
    def __init__(self, d: int, hidden: int = 100, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(d, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden
        self.fc = nn.Linear(hidden, 2)

    def features(self, x):
        out, (h, c) = self.rnn(x)
        return self.drop(h[-1])

    def forward(self, x):
        return self.fc(self.features(x))


class GRUBackbone(nn.Module):
    def __init__(self, d: int, hidden: int = 100, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.GRU(d, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden
        self.fc = nn.Linear(hidden, 2)

    def features(self, x):
        out, h = self.rnn(x)
        return self.drop(h[-1])

    def forward(self, x):
        return self.fc(self.features(x))


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dropout=0.2):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class FCNNBackbone(nn.Module):
    """3 ConvBlocks (Conv1d+BN+ReLU+Dropout), Table I."""
    def __init__(self, d: int, channels=(128, 256, 128), dropout: float = 0.2):
        super().__init__()
        ks = [3, 3, 3]
        blocks = []
        c_in = d
        for c_out, k in zip(channels, ks):
            blocks.append(ConvBlock(c_in, c_out, k, dropout))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c_in
        self.fc = nn.Linear(c_in, 2)

    def features(self, x):
        h = x.transpose(1, 2)  # (B, d, k)
        h = self.blocks(h)
        return self.pool(h).squeeze(-1)

    def forward(self, x):
        return self.fc(self.features(x))


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dropout=0.2):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.short = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.drop(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.short(x))


class ResNetBackbone(nn.Module):
    """3 ResBlocks, Table I."""
    def __init__(self, d: int, channels=(64, 128, 128), dropout: float = 0.2):
        super().__init__()
        blocks, c_in = [], d
        for c_out in channels:
            blocks.append(ResBlock(c_in, c_out, k=3, dropout=dropout))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c_in
        self.fc = nn.Linear(c_in, 2)

    def features(self, x):
        h = x.transpose(1, 2)
        h = self.blocks(h)
        return self.pool(h).squeeze(-1)

    def forward(self, x):
        return self.fc(self.features(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TSTBackbone(nn.Module):
    """3 TSTEncoderLayers (multihead attn+FFN, GELU), Table I."""
    def __init__(self, d: int, d_model: int = 64, nhead: int = 8, layers: int = 3,
                 dim_ff: int = 128, dropout: float = 0.1, seq_len: int = 6):
        super().__init__()
        self.proj = nn.Linear(d, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len + 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_dim = d_model
        self.fc = nn.Linear(d_model, 2)

    def features(self, x):
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        return h.mean(dim=1)

    def forward(self, x):
        return self.fc(self.features(x))


BACKBONES = {
    "LSTM": LSTMBackbone, "GRU": GRUBackbone, "FCNN": FCNNBackbone,
    "ResNet": ResNetBackbone, "TST": TSTBackbone,
}


def build_backbone(name: str, d: int, seq_len: int = 6):
    cls = BACKBONES[name]
    if name == "TST":
        return cls(d, seq_len=seq_len)
    return cls(d)
