"""
Publication-quality figure generation (master task Sec. 31), all reading
from saved CSV/JSON result files -- no numbers are invented or typed by
hand. Uses matplotlib only (no seaborn dependency) with a consistent style.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_benchmark_bars(df: pd.DataFrame, metric: str, out_path: str,
                        method_col: str = "method", highlight: str | None = None):
    df = df.sort_values(metric, ascending=True)
    colors = ["#d62728" if highlight and m == highlight else "#4c72b0" for m in df[method_col]]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(df))))
    ax.barh(df[method_col], df[metric], color=colors)
    ax.set_xlabel(metric)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_pareto(df: pd.DataFrame, x_col: str, y_col: str, out_path: str,
               label_col: str = "method", highlight: str | None = None, x_log: bool = True):
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in df.iterrows():
        is_h = highlight and row[label_col] == highlight
        ax.scatter(row[x_col], row[y_col], s=90 if is_h else 45,
                   c="#d62728" if is_h else "#4c72b0", zorder=3 if is_h else 2)
        ax.annotate(row[label_col], (row[x_col], row[y_col]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    if x_log:
        ax.set_xscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_ablation_bars(ablation_json: dict, metric: str, out_path: str, order=None):
    names = order or list(ablation_json.keys())
    means = [ablation_json[n]["agg"][f"{metric}_mean"] for n in names]
    stds = [ablation_json[n]["agg"][f"{metric}_std"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#d62728" if n == "Full" else "#4c72b0" for n in names]
    ax.bar(names, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_robustness_lines(sweep: dict, metric: str, out_path: str, xlabel: str = "perturbation level"):
    xs, ys = [], []
    for k, v in sweep.items():
        xs.append(float(k.split("_")[-1].replace("x", "")))
        ys.append(v[metric])
    order = np.argsort(xs)
    xs, ys = np.array(xs)[order], np.array(ys)[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", color="#4c72b0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_path: str):
    """curves: {name: (fpr_array, tpr_array)}."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_architecture_placeholder_free(*args, **kwargs):
    raise NotImplementedError(
        "Architecture diagrams (Fig. 1) are authored directly (e.g. as a "
        "vector drawing) once the final model is frozen, not generated from "
        "result CSVs -- intentionally not implemented here.")
