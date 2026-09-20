"""
SOURCE_METHOD_REPRODUCTION driver.

For a given system, for each of the 5 backbones:
  1. Train the basic BDD-passing-FDIA detector (Table II "FDIA" row).
  2. Evaluate it on held-out BDD-passing FDIA (clean) and on freshly-crafted
     C&W adversarial FDIAs from the TEST split (Table II "advFDIA" row).
  3. Craft C&W adversarial FDIAs from the dedicated adv-train pool (disjoint
     bus set) and use them (label=1) to adversarially retrain a "Robust"
     variant (Table III).
  4. Evaluate the robust variant both in-distribution (adv examples from the
     same adv-train bus set) and out-of-distribution (adv examples crafted
     against the four canonical test-case bus sets) -> reproduces the
     paper's Table III generalization-failure finding.
  5. Train the multiheaded ensemble + LLLA on top of the frozen basic
     backbone, and the MC-dropout / deep-ensemble UQ baselines, then report
     FPR(TPR95)/FNR(TNR95)/D-Error/AUROC (Table IV) using the test split's
     clean vs. adversarial-FDIA sequences.

All numbers are written to results/reproduction/{system}_reproduction.json
as our own "Local reproduction" values -- see REPRODUCTION_REPORT.md for the
side-by-side comparison against the paper's "Published result" numbers.
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system, SeqDataset
from models.backbones import build_backbone
from models.uncertainty import (MultiHeadEnsemble, multihead_loss, LLLAMultiHeadEnsemble,
                                 mc_dropout_mi, deep_ensemble_mi, mutual_information)
from sim.network import build_network, dc_jacobian
from attacks.cw_attack import craft_adversarial_fdia
from train.common import set_seed, train_classifier, evaluate_classifier, binary_metrics, uq_detection_metrics
import torch.nn.functional as F

BACKBONE_NAMES = ["LSTM", "GRU", "FCNN", "ResNet", "TST"]


def to_loader(X, y, bs=128, shuffle=False):
    return DataLoader(SeqDataset(X, y), batch_size=bs, shuffle=shuffle)


def craft_cw_batch(model, X, buses_list, jac, device, bs=64, C_grid=(1.0, 10.0), n_iter=300, lr=0.01):
    n_branch, n_bus = jac["n_branch"], jac["n_bus"]
    all_adv, all_success = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32)
        bb = buses_list[i:i + bs]
        Z_adv, success, norm, c = craft_adversarial_fdia(
            model, xb, bb, jac["H_dc"], jac["non_slack"], n_branch, n_bus, device,
            C_grid=C_grid, n_iter=n_iter, lr=lr)
        all_adv.append(Z_adv.cpu().numpy())
        all_success.append(success.cpu().numpy())
    return np.concatenate(all_adv), np.concatenate(all_success)


def run_system(system: str, seed: int = 0, epochs: int = 25, cw_iter: int = 300, out_dir="results/reproduction"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    data = load_system(system)
    meta = data["meta"]
    d = meta["d"]
    net = build_network(system)
    jac = dc_jacobian(net)

    train_loader = to_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader = to_loader(data["X_val"], data["y_val"])
    test_loader = to_loader(data["X_test"], data["y_test"])

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"models_ckpt/{system}", exist_ok=True)
    report_path = f"{out_dir}/{system}_reproduction.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        print(f"[{system}] resuming: {list(report['backbones'].keys())} already done")
    else:
        report = {"system": system, "seed": seed, "backbones": {}}

    adv_pool_X = data["adv_pool_X"]
    adv_pool_buses = [meta["adv_train_buses"]] * len(adv_pool_X)

    case_names = [k[5:-2] for k in data if k.startswith("case_") and k.endswith("_X")]

    for name in BACKBONE_NAMES:
        if name in report["backbones"]:
            print(f"[{system}] === {name} === (skipping, already done)")
            continue
        t0 = time.time()
        print(f"[{system}] === {name} ===")
        model = build_backbone(name, d, seq_len=6)
        model, val_f1 = train_classifier(model, train_loader, val_loader, device, epochs=epochs)
        train_time = time.time() - t0

        clean_metrics, clean_probs, clean_y = evaluate_classifier(model, test_loader, device)
        print(f"  basic test macro_f1={clean_metrics['macro_f1']:.4f} auroc={clean_metrics['roc_auc']:.4f} (train {train_time:.1f}s)")

        # C&W adversarial FDIAs crafted from TEST-split positives (Table II "advFDIA")
        test_pos_idx = np.where(data["y_test"] == 1)[0]
        test_pos_X = data["X_test"][test_pos_idx]
        # bus labels unknown for generic test positives (only case-eval sets carry ground truth);
        # use adv-train-bus-like generic small random supports consistent with our generic pool.
        # For a controlled, ground-truth-bearing "advFDIA" set we instead use the 4 canonical cases.
        case_adv_results = {}
        case_adv_Z = {}  # cache: reused below for the UQ report (same basic `model`)
        for case in case_names:
            Xc = data[f"case_{case}_X"]
            buses_c = data[f"case_{case}_buses"].tolist()
            buses_list = [buses_c] * len(Xc)
            t1 = time.time()
            Z_adv, success = craft_cw_batch(model, Xc, buses_list, jac, device, n_iter=cw_iter)
            cw_time = time.time() - t1
            y_adv = np.ones(len(Xc), dtype=np.int64)
            m_adv, _, _ = evaluate_classifier(model, to_loader(Z_adv, y_adv), device)
            case_adv_results[case] = dict(
                n=len(Xc), attack_success_rate=float(success.mean()),
                basic_acc_on_adv=1 - m_adv["fnr"], cw_time_sec=cw_time)
            case_adv_Z[case] = Z_adv
            print(f"  case {case}: CW success={success.mean():.2f} basic_acc_on_adv={1-m_adv['fnr']:.3f}")

        # --- Adversarial training (Table III) ---
        t2 = time.time()
        adv_examples, adv_success = craft_cw_batch(model, adv_pool_X, adv_pool_buses, jac, device, n_iter=cw_iter)
        adv_examples = adv_examples[adv_success]
        cw_advtrain_time = time.time() - t2
        n_adv = len(adv_examples)
        if n_adv > 0:
            X_robust = np.concatenate([data["X_train"], adv_examples], axis=0)
            y_robust = np.concatenate([data["y_train"], np.ones(n_adv, dtype=np.int64)], axis=0)
        else:
            X_robust, y_robust = data["X_train"], data["y_train"]
        robust_model = build_backbone(name, d, seq_len=6)
        robust_model.load_state_dict(model.state_dict())
        robust_model, robust_val_f1 = train_classifier(
            robust_model, to_loader(X_robust, y_robust, shuffle=True), val_loader, device, epochs=max(8, epochs // 2))

        robust_case_results = {}
        for case in case_names:
            Xc = data[f"case_{case}_X"]
            buses_c = data[f"case_{case}_buses"].tolist()
            in_dist = set(buses_c) == set(meta["adv_train_buses"])
            Z_adv, success = craft_cw_batch(robust_model, Xc, [buses_c] * len(Xc), jac, device, n_iter=cw_iter)
            y_adv = np.ones(len(Xc), dtype=np.int64)
            m_adv, _, _ = evaluate_classifier(robust_model, to_loader(Z_adv, y_adv), device)
            robust_case_results[case] = dict(acc_on_adv=1 - m_adv["fnr"], attack_success_rate=float(success.mean()))
        # in-distribution robustness check: attack the adv-train bus set itself
        Z_adv_id, success_id = craft_cw_batch(robust_model, adv_pool_X, adv_pool_buses, jac, device, n_iter=cw_iter)
        m_id, _, _ = evaluate_classifier(robust_model, to_loader(Z_adv_id, np.ones(len(Z_adv_id), dtype=np.int64)), device)
        robust_case_results["adv_train_bus_set_in_distribution"] = dict(acc_on_adv=1 - m_id["fnr"])

        torch.save(model.state_dict(), f"models_ckpt/{system}/{name}_basic.pt")
        torch.save(robust_model.state_dict(), f"models_ckpt/{system}/{name}_robust.pt")

        # --- Multiheaded ensemble + LLLA (proposed method) + MC-dropout / deep-ensemble baselines ---
        model.eval()
        ens = MultiHeadEnsemble(model, m=5, hidden=64).to(device)
        opt = torch.optim.Adam(ens.heads.parameters(), lr=1e-3)
        for ep in range(15):
            ens.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits_list = ens(xb)
                loss, _ = multihead_loss(logits_list, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ens.heads.parameters(), max_norm=5.0)
                opt.step()

        llla = LLLAMultiHeadEnsemble(ens, prior_precision=5.0)
        llla.fit(train_loader, device)

        # Deep-ensemble UQ baseline (source paper Table IV): 4 additional
        # independently-initialized/-trained members + the basic `model`.
        de_members = [model]
        for j in range(4):
            set_seed(seed * 1000 + j + 1)
            m_j = build_backbone(name, d, seq_len=6)
            m_j, _ = train_classifier(m_j, train_loader, val_loader, device, epochs=max(8, epochs // 2))
            de_members.append(m_j.to(device))

        def mi_scores(X, model_forward):
            loader = to_loader(X, np.zeros(len(X), dtype=np.int64))
            out = []
            for xb, _ in loader:
                xb = xb.to(device)
                out.append(model_forward(xb).cpu().numpy())
            return np.concatenate(out)

        uq_report = {}
        for uq_name, fn in [
            ("proposed_llla", lambda xb: llla.mi(xb, mc_samples=20)),
            ("mc_dropout", lambda xb: mc_dropout_mi(model, xb, T=50)),
            ("deep_ensemble", lambda xb: deep_ensemble_mi(de_members, xb)),
        ]:
            mi_clean = mi_scores(data["X_test"], fn)
            case_mi = {}
            for case in case_names:
                Z_adv = case_adv_Z[case]  # reuse: same basic `model`, already crafted above
                mi_adv = mi_scores(Z_adv, fn)
                labels = np.concatenate([np.zeros(len(mi_clean)), np.ones(len(mi_adv))])
                scores = np.concatenate([mi_clean, mi_adv])
                case_mi[case] = uq_detection_metrics(labels, scores)
            uq_report[uq_name] = case_mi

        report["backbones"][name] = dict(
            train_time_sec=train_time, val_macro_f1=val_f1,
            test_clean_metrics=clean_metrics,
            case_adv_results=case_adv_results,
            robust_val_macro_f1=robust_val_f1,
            robust_case_results=robust_case_results,
            n_adv_train_examples=int(n_adv),
            uq_report=uq_report,
            params=sum(p.numel() for p in model.parameters()),
        )
        with open(f"{out_dir}/{system}_reproduction.json", "w") as f:
            json.dump(report, f, indent=2, default=float)

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="case30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--cw_iter", type=int, default=300)
    args = ap.parse_args()
    run_system(args.system, seed=args.seed, epochs=args.epochs, cw_iter=args.cw_iter)
