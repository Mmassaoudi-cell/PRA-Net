"""
Builds BENCHMARK_WTL.csv (master task Sec. 30): PRA-Net (5 seeds) vs. every
benchmark, per system, on macro-F1 (the common metric across all methods).
Since the classical/boosting/deep benchmark suite (run_benchmarks.py) is
single-run (one seed) -- a standard practical compromise for a suite this
large -- the comparison is a one-sample test of whether PRA-Net's 5-seed
distribution's 95% CI excludes the benchmark's point estimate, rather than a
two-sample paired test (reserved for candidate-vs-candidate comparisons in
MODEL_SELECTION_REPORT.md, where every side has multiple seeds).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats


def magnitude_bucket(abs_diff: float) -> str:
    if abs_diff < 0.005:
        return "negligible"
    if abs_diff < 0.02:
        return "small"
    if abs_diff < 0.10:
        return "moderate"
    return "large"


def classify_vs_point(seed_values: np.ndarray, baseline_point: float, margin: float = 0.005) -> dict:
    mean = float(np.mean(seed_values))
    std = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
    n = len(seed_values)
    se = std / np.sqrt(n) if n > 1 else 0.0
    # one-sample t-test: H0: mean(seed_values) == baseline_point
    if n > 1 and se > 0:
        t_stat, p_value = stats.ttest_1samp(seed_values, baseline_point)
    else:
        t_stat, p_value = float("nan"), 1.0
    diff = mean - baseline_point
    # One-sample effect size (Cohen's d using PRA-Net's own seed spread as
    # the standardizer, since the single-run benchmark contributes no
    # variance estimate). Statistical significance alone can be driven
    # purely by PRA-Net's very small seed variance, so this magnitude field
    # is reported alongside p-value to let a reader separate a trivial
    # margin (e.g. 0.0009 macro-F1) from a substantial one.
    effect_size = diff / std if std > 0 else float("inf") if diff != 0 else 0.0
    magnitude = magnitude_bucket(abs(diff))
    if not np.isnan(p_value) and p_value < 0.05:
        outcome = "win" if diff > 0 else "loss"
        verdict = "statistically_superior" if diff > 0 else "inferior"
    else:
        if diff > margin:
            outcome, verdict = "win", "statistically_indistinguishable"
        elif diff < -margin:
            outcome, verdict = "loss", "statistically_indistinguishable"
        else:
            outcome, verdict = "tie", "statistically_indistinguishable"
    return dict(proposed_mean=mean, baseline=baseline_point, mean_diff=diff,
                effect_size_cohens_d=effect_size, magnitude=magnitude,
                p_value=float(p_value), outcome=outcome, verdict=verdict)


def build(systems=("case30", "case118"), out_path="BENCHMARK_WTL.csv"):
    rows = []
    for system in systems:
        bench_df = pd.read_csv(f"results/benchmarks/{system}_benchmarks.csv")
        with open(f"results/benchmarks/{system}_final_pranet_test.json") as f:
            pranet_seeds = json.load(f)
        pranet_f1 = np.array([s["clean_detection"]["macro_f1"] for s in pranet_seeds])

        for _, row in bench_df.iterrows():
            cls = classify_vs_point(pranet_f1, row["macro_f1"])
            rows.append(dict(dataset=system, benchmark=row["method"], category=row["category"],
                              metric="macro_f1", **cls))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    summary = df["outcome"].value_counts().to_dict()
    print(f"Proposed (PRA-Net): {summary.get('win',0)} wins / {summary.get('tie',0)} ties / "
          f"{summary.get('loss',0)} losses (out of {len(df)} comparisons)")
    return df


if __name__ == "__main__":
    build()
