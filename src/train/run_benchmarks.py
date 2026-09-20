"""
Full benchmark-suite driver (master task Sec. 14): classical + boosting +
deep + graph baselines, trained on TRAIN, selected on VAL where applicable,
reported on TEST -- all under the shared metric set (macro-F1 primary,
balanced accuracy / minority recall / PR-AUC / ROC-AUC / FPR / FNR
secondary). The five source-paper backbones' plain-supervised numbers
(already computed in run_reproduction.py) and the SOURCE_METHOD_REPRODUCTION
proposed-LLLA detector are merged in from results/reproduction/*.json rather
than recomputed.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system
from sim.network import build_network, dc_jacobian
from baselines.classical import build_classical_suite
from baselines.deep import TCNBackbone, GCNBenchmark
from train.common import set_seed, train_classifier, evaluate_classifier, binary_metrics


def to_loader(X, y, bs=128, shuffle=False):
    from torch.utils.data import DataLoader
    from utils.data import SeqDataset
    return DataLoader(SeqDataset(X, y), batch_size=bs, shuffle=shuffle)


def run_classical(system, data, seed=0):
    rows = []
    suite = build_classical_suite(seed)
    for name, wrapper in suite.items():
        t0 = time.time()
        wrapper.fit(data["X_train"], data["y_train"])
        train_time = time.time() - t0
        prob_test = wrapper.predict_proba(data["X_test"])
        m = binary_metrics(data["y_test"], prob_test)
        rows.append(dict(method=name, category="classical_or_boosting", train_time_sec=train_time,
                          params=wrapper.n_params(), **m))
        print(f"[bench] {name}: macro_f1={m['macro_f1']:.4f} auroc={m['roc_auc']:.4f} time={train_time:.1f}s")
    return rows


def run_deep(system, data, jac, device, epochs=25, seed=0):
    rows = []
    d = data["meta"]["d"]
    train_loader = to_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader = to_loader(data["X_val"], data["y_val"])
    test_loader = to_loader(data["X_test"], data["y_test"])

    for name, builder in [
        ("TCN", lambda: TCNBackbone(d)),
        ("GCN", lambda: GCNBenchmark(jac["n_branch"], jac["n_bus"], jac["branches"])),
    ]:
        set_seed(seed)
        t0 = time.time()
        model = builder()
        model, val_f1 = train_classifier(model, train_loader, val_loader, device, epochs=epochs)
        train_time = time.time() - t0
        m, probs, ys = evaluate_classifier(model, test_loader, device)
        rows.append(dict(method=name, category="deep_or_graph", train_time_sec=train_time,
                          params=sum(p.numel() for p in model.parameters()), **m))
        print(f"[bench] {name}: macro_f1={m['macro_f1']:.4f} auroc={m['roc_auc']:.4f} time={train_time:.1f}s")
    return rows


def merge_reproduction_rows(system, repro_path):
    rows = []
    if not os.path.exists(repro_path):
        return rows
    with open(repro_path) as f:
        r = json.load(f)
    for name, v in r["backbones"].items():
        m = v["test_clean_metrics"]
        rows.append(dict(method=f"{name} (plain supervised)", category="source_paper_backbone",
                          train_time_sec=v["train_time_sec"], params=v["params"], **m))
    return rows


def run_all(system: str, seed: int = 0, epochs: int = 25, out_dir: str = "results/benchmarks"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    rows += run_classical(system, data, seed)
    rows += run_deep(system, data, jac, device, epochs, seed)
    rows += merge_reproduction_rows(system, f"results/reproduction/{system}_reproduction.json")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/{system}_benchmarks.csv", index=False)
    print(df[["method", "category", "macro_f1", "roc_auc", "fpr", "fnr"]].to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="case30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    run_all(args.system, args.seed, args.epochs)
