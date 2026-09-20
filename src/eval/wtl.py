"""
Win/Tie/Loss analysis (master task Sec. 30): compares the proposed final
model against each benchmark, per dataset/system, using paired-seed
statistics (eval/statistics.py) to classify each comparison and writes
BENCHMARK_WTL.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from eval.statistics import paired_comparison, verdict


def classify(proposed_scores: np.ndarray, baseline_scores: np.ndarray,
             practical_margin: float = 0.01) -> dict:
    """proposed_scores/baseline_scores: per-seed metric values (e.g. macro-F1)
    on the same dataset/system. Returns win/tie/loss + statistical detail."""
    comp = paired_comparison(proposed_scores, baseline_scores)
    stat_verdict = verdict(comp.get("wilcoxon_p", comp.get("ttest_p", 1.0)), comp["mean_diff"])
    if stat_verdict == "statistically_superior":
        outcome = "win"
    elif stat_verdict == "inferior":
        outcome = "loss"
    else:
        # statistically indistinguishable -> fall back to a practical margin
        if comp["mean_diff"] > practical_margin:
            outcome = "win"
        elif comp["mean_diff"] < -practical_margin:
            outcome = "loss"
        else:
            outcome = "tie"
    return dict(outcome=outcome, verdict=stat_verdict, **comp)


def build_wtl_table(records: list[dict]) -> pd.DataFrame:
    """records: list of dicts with keys dataset, benchmark, proposed_scores
    (list/array of per-seed values), baseline_scores (same)."""
    rows = []
    for r in records:
        cls = classify(np.asarray(r["proposed_scores"]), np.asarray(r["baseline_scores"]))
        rows.append(dict(dataset=r["dataset"], benchmark=r["benchmark"], metric=r.get("metric", "macro_f1"),
                          proposed_mean=float(np.mean(r["proposed_scores"])),
                          baseline_mean=float(np.mean(r["baseline_scores"])), **cls))
    df = pd.DataFrame(rows)
    return df


def summarize_wtl(df: pd.DataFrame) -> dict:
    counts = df["outcome"].value_counts().to_dict()
    return dict(wins=int(counts.get("win", 0)), ties=int(counts.get("tie", 0)),
                losses=int(counts.get("loss", 0)))
