"""Shared training utilities: fit loop with early stopping, metric computation."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                              average_precision_score, recall_score, roc_curve, precision_score)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_metrics(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    out = dict(
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        balanced_acc=balanced_accuracy_score(y_true, y_pred),
        minority_recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        accuracy=(y_pred == y_true).mean(),
    )
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    out["fpr"] = fp / max(fp + tn, 1)
    out["fnr"] = fn / max(fn + tp, 1)
    return out


def uq_detection_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Source-paper Table IV metrics for a threshold-free uncertainty score:
    FPR(TPR95), FNR(TNR95), D-Error (min avg FPR/FNR over thresholds), AUROC."""
    if len(np.unique(y_true)) < 2:
        return dict(fpr_tpr95=float("nan"), fnr_tnr95=float("nan"),
                    d_error=float("nan"), auroc=float("nan"))
    if not np.all(np.isfinite(scores)):
        n_bad = (~np.isfinite(scores)).sum()
        print(f"  [warn] uq_detection_metrics: {n_bad}/{len(scores)} non-finite scores "
              f"(unstable UQ member) -- clipping to finite range")
        finite = scores[np.isfinite(scores)]
        fallback = float(np.median(finite)) if len(finite) else 0.0
        scores = np.where(np.isfinite(scores), scores, fallback)
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    tnr = 1 - fpr
    idx_tpr95 = np.argmin(np.abs(tpr - 0.95))
    idx_tnr95 = np.argmin(np.abs(tnr - 0.95))
    d_error = float(np.min((fpr + fnr) / 2))
    auroc = float(roc_auc_score(y_true, scores))
    return dict(fpr_tpr95=float(fpr[idx_tpr95]), fnr_tnr95=float(fnr[idx_tnr95]),
                d_error=d_error, auroc=auroc)


def localization_metrics(y_true_bus: np.ndarray, y_prob_bus: np.ndarray, thresh: float = 0.5) -> dict:
    """Multi-label bus-localization metrics (master-task Sec. 17): treats
    each bus as a one-vs-rest label and macro-averages across buses --
    replacing the source paper's per-bus *binary* recall/specificity
    evaluation with genuine multiclass/multi-label classification metrics."""
    y_pred_bus = (y_prob_bus >= thresh).astype(int)
    out = dict(
        macro_f1=f1_score(y_true_bus, y_pred_bus, average="macro", zero_division=0),
        micro_f1=f1_score(y_true_bus, y_pred_bus, average="micro", zero_division=0),
        macro_precision=precision_score(y_true_bus, y_pred_bus, average="macro", zero_division=0),
        macro_recall=recall_score(y_true_bus, y_pred_bus, average="macro", zero_division=0),
    )
    try:
        out["macro_pr_auc"] = average_precision_score(y_true_bus, y_prob_bus, average="macro")
    except ValueError:
        out["macro_pr_auc"] = float("nan")
    return out


def train_classifier(model, train_loader, val_loader, device, epochs=30, lr=1e-3,
                      patience=6, weight_decay=1e-5, verbose=False):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1, best_state, bad_epochs = -1, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                p = F.softmax(model(xb), dim=-1)[:, 1].cpu().numpy()
                probs.append(p)
                ys.append(yb.numpy())
        probs, ys = np.concatenate(probs), np.concatenate(ys)
        m = binary_metrics(ys, probs)
        if verbose:
            print(f"  epoch {ep}: val_macro_f1={m['macro_f1']:.4f} val_auroc={m['roc_auc']:.4f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = F.softmax(model(xb), dim=-1)[:, 1].cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    return binary_metrics(ys, probs), probs, ys
