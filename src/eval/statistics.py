"""
Statistical validation utilities (master task Sec. 22): paired comparisons
across seeds with Wilcoxon signed-rank / paired t-test, Holm-Bonferroni
correction across multiple comparisons, and effect size (Cohen's d for
paired samples).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean, std = values.mean(), values.std(ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * se
    return dict(mean=float(mean), std=float(std), median=float(np.median(values)),
                ci95_lo=float(mean - ci95), ci95_hi=float(mean + ci95), n=n)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def paired_comparison(proposed: np.ndarray, baseline: np.ndarray) -> dict:
    proposed, baseline = np.asarray(proposed), np.asarray(baseline)
    diff = proposed - baseline
    out = dict(mean_diff=float(diff.mean()), effect_size=cohens_d_paired(proposed, baseline))
    if len(proposed) >= 2 and not np.allclose(proposed, baseline):
        try:
            t_stat, t_p = stats.ttest_rel(proposed, baseline)
        except Exception:
            t_stat, t_p = float("nan"), float("nan")
        try:
            w_stat, w_p = stats.wilcoxon(proposed, baseline)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        out.update(ttest_stat=float(t_stat), ttest_p=float(t_p),
                    wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p))
    else:
        out.update(ttest_p=1.0, wilcoxon_p=1.0)
    return out


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns per-comparison reject-H0 flags."""
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if p_values[idx] <= thresh:
            reject[idx] = True
        else:
            break  # step-down: once one fails, all remaining (larger p) fail too
    return reject


def verdict(p_value: float, mean_diff: float, alpha: float = 0.05) -> str:
    if np.isnan(p_value):
        return "statistically_indistinguishable"
    if p_value <= alpha:
        return "statistically_superior" if mean_diff > 0 else "inferior"
    return "statistically_indistinguishable" if mean_diff >= 0 else "inferior"
