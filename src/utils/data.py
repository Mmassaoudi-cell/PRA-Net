"""Dataset loading utilities shared across reproduction, baselines and candidates."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class JointSeqDataset(Dataset):
    """(X, y_detection, y_bus_localization) triples for candidates with a
    jointly-trained detection + per-bus multi-label localization head."""
    def __init__(self, X: np.ndarray, y: np.ndarray, ybus: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.ybus = torch.tensor(ybus, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ybus[idx]


class Standardizer:
    """Fit on TRAIN split only (no leakage), per-channel z-score."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X_train: np.ndarray):
        self.mean = X_train.mean(axis=(0, 1), keepdims=True)
        self.std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def multihot_bus_labels(attacked_buses_col: pd.Series, n_bus: int) -> np.ndarray:
    """Parse the manifest's ';'-joined attacked-bus-id strings into an
    (n_seq, n_bus) multi-label 0/1 matrix (row order == array row order,
    since the manifest is written in the same per-sequence loop as X/y)."""
    Y = np.zeros((len(attacked_buses_col), n_bus), dtype=np.float32)
    for i, s in enumerate(attacked_buses_col.values):
        if isinstance(s, str) and s:
            for b in s.split(";"):
                Y[i, int(b)] = 1.0
    return Y


def load_system(system: str, data_root: str = "data/processed"):
    d = np.load(f"{data_root}/{system}/dataset.npz")
    with open(f"{data_root}/{system}/meta.json") as f:
        meta = json.load(f)
    X, y, split = d["X"], d["y"], d["split"]
    manifest = pd.read_csv(f"{data_root}/{system}/DATA_SPLIT_MANIFEST.csv")
    assert len(manifest) == len(X), "manifest/array row-count mismatch"
    Ybus = multihot_bus_labels(manifest["attacked_buses"], meta["n_bus"])

    out = {}
    for s in ["train", "val", "test"]:
        mask = split == s
        out[f"X_{s}"] = X[mask]
        out[f"y_{s}"] = y[mask]
        out[f"ybus_{s}"] = Ybus[mask]

    scaler = Standardizer().fit(out["X_train"])
    for s in ["train", "val", "test"]:
        out[f"X_{s}"] = scaler.transform(out[f"X_{s}"]).astype(np.float32)

    out["scaler"] = scaler
    out["meta"] = meta
    out["adv_pool_X"] = scaler.transform(d["adv_pool_X"]).astype(np.float32)

    n_bus = meta["n_bus"]
    adv_buses = meta["adv_train_buses"]
    out["adv_pool_ybus"] = np.tile(
        np.isin(np.arange(n_bus), adv_buses).astype(np.float32), (len(d["adv_pool_X"]), 1))

    for key in d.files:
        if key.startswith("case_eval_") and key.endswith("_X"):
            case = key[len("case_eval_"):-2]
            out[f"case_{case}_X"] = scaler.transform(d[key]).astype(np.float32)
            buses_c = d[f"case_eval_{case}_buses"]
            out[f"case_{case}_buses"] = buses_c
            out[f"case_{case}_ybus"] = np.tile(
                np.isin(np.arange(n_bus), buses_c).astype(np.float32), (len(d[key]), 1))
    return out
