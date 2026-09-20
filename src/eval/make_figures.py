"""Generates the data-driven figures (master task Sec. 31, Figs. 3/5/6) from
saved result files. Figures 1/2/4 (architecture / conceptual-workflow /
difficult-case diagrams) are authored directly as vector drawings, not
generated from CSVs -- see manuscript/figures/ for those."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.figures import fig_benchmark_bars, fig_pareto, fig_ablation_bars, fig_roc_curves
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def fig3_main_benchmark():
    df30 = pd.read_csv("results/benchmarks/case30_benchmarks.csv")
    with open("results/benchmarks/case30_final_pranet_test.json") as f:
        pranet = json.load(f)
    pranet_row = dict(method="PRA-Net (proposed)", category="proposed",
                       macro_f1=np.mean([s["clean_detection"]["macro_f1"] for s in pranet]))
    df = pd.concat([df30[["method", "category", "macro_f1"]], pd.DataFrame([pranet_row])], ignore_index=True)
    fig_benchmark_bars(df, "macro_f1", f"{OUT}/fig3_main_benchmark_case30.png",
                        highlight="PRA-Net (proposed)")


def fig4_localization_comparison():
    """Difficult-case figure: localization macro-F1, PRA-Net vs. source method,
    per canonical test case, both systems."""
    with open("results/reproduction/case30_localization_reproduction.json") as f:
        loc30 = json.load(f)
    with open("results/reproduction/case118_localization_reproduction.json") as f:
        loc118 = json.load(f)
    with open("results/benchmarks/case30_final_pranet_test.json") as f:
        pranet30 = json.load(f)
    with open("results/benchmarks/case118_final_pranet_test.json") as f:
        pranet118 = json.load(f)

    labels, source_vals, pranet_vals = [], [], []
    for sysname, locrepro, pranet in [("case30", loc30, pranet30), ("case118", loc118, pranet118)]:
        for case, d in locrepro["cases"].items():
            if "macro_f1" not in d:
                continue
            labels.append(f"{sysname}\n{case}")
            source_vals.append(d["macro_f1"])
        # PRA-Net "clean" test-set localization (natural distribution, not the
        # adversarial-only comparison, since the source method's own number
        # above is also computed on its BDD-passing/adversarial mix) --
        # report the mean across seeds as the comparison bar.
        pranet_vals.append(np.mean([s["clean_localization"]["macro_f1"] for s in pranet]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, source_vals, width=0.5, label="Source method (VAE-search + gradient attribution)", color="#4c72b0")
    ax.axhline(y=np.mean(pranet_vals), color="#d62728", linestyle="--", label="PRA-Net (proposed), mean over test set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Localization macro-F1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_localization_comparison.png")
    plt.close(fig)


def fig5_ablation():
    with open("results/ablation/ablation_results.json") as f:
        abl = json.load(f)
    order = ["Full", "-Physics", "-AttnBias", "-Attention", "-JointLoc", "Plain"]
    fig_ablation_bars(abl, "loc_macro_f1", f"{OUT}/fig5_ablation_localization.png", order=order)
    fig_ablation_bars(abl, "macro_f1", f"{OUT}/fig5b_ablation_detection.png", order=order)


def fig6_pareto():
    df30 = pd.read_csv("results/benchmarks/case30_benchmarks.csv")
    with open("results/benchmarks/case30_final_pranet_test.json") as f:
        pranet = json.load(f)
    pranet_row = dict(method="PRA-Net (proposed)",
                       macro_f1=np.mean([s["clean_detection"]["macro_f1"] for s in pranet]),
                       params=pranet[0]["params"])
    # curate a representative, non-overlapping subset for readability (kNN's
    # "params" count is its stored training set, not learned weights, and is
    # excluded from this plot for that reason; full table in the CSV/appendix)
    keep = ["LogReg", "GCN", "TCN", "XGBoost", "LightGBM", "MLP",
            "HistGB", "RandomForest", "ExtraTrees", "TST (plain supervised)"]
    df30 = df30[df30["method"].isin(keep)]
    df = pd.concat([df30[["method", "macro_f1", "params"]], pd.DataFrame([pranet_row])], ignore_index=True)
    df = df[df["params"] > 0]
    fig_pareto(df, "params", "macro_f1", f"{OUT}/fig6_pareto_params.png",
               highlight="PRA-Net (proposed)")


def fig7_adversarial_diagnostics():
    """Gradient-norm trajectory (PRA-Net vs. LSTM) and transfer-attack
    non-transferability, addressing the gradient-masking / false-robustness
    concern with direct empirical evidence."""
    with open("results/robustness/case30_adversarial_diagnostics.json") as f:
        d30 = json.load(f)
    with open("results/robustness/case118_adversarial_diagnostics.json") as f:
        d118 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    labels, pranet_gn, lstm_gn = [], [], []
    for sysname, d in [("case30", d30), ("case118", d118)]:
        for case, c in d["cases"].items():
            labels.append(f"{sysname}\n{case}")
            pranet_gn.append(c["gradient_norm_pranet_mean"])
            lstm_gn.append(c["gradient_norm_lstm_mean"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, pranet_gn, width=0.4, label="PRA-Net", color="#d62728")
    ax.bar(x + 0.2, lstm_gn, width=0.4, label="LSTM (source backbone)", color="#4c72b0")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean attack-loss gradient norm\n(log scale)")
    ax.legend(fontsize=8)
    ax.set_title("(a) Gradient-norm trajectory")

    ax = axes[1]
    fools_fcnn, transfers_label, transfers_u = [], [], []
    for sysname, d in [("case30", d30), ("case118", d118)]:
        for case, c in d["cases"].items():
            fools_fcnn.append(c["transfer_fools_fcnn_rate"] * 100)
            transfers_label.append(c["transfer_fools_pranet_classlabel_rate"] * 100)
            transfers_u.append(c["transfer_evades_pranet_uncertainty_rate"] * 100)
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, fools_fcnn, width=w, label="Fools surrogate (FCNN)", color="#4c72b0")
    ax.bar(x, transfers_label, width=w, label="Transfers to PRA-Net (label)", color="#dd8452")
    ax.bar(x + w, transfers_u, width=w, label="Transfers to PRA-Net (evades $u$)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=7)
    ax.set_title("(b) Black-box transfer attack")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_adversarial_diagnostics.png")
    plt.close(fig)


if __name__ == "__main__":
    fig3_main_benchmark()
    fig4_localization_comparison()
    fig5_ablation()
    fig6_pareto()
    fig7_adversarial_diagnostics()
    print("Figures written to", OUT)
