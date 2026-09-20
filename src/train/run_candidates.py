"""
Candidate screening driver (master task Sec. 9): Stage 1 smoke test (1 seed,
reduced budget) -> Stage 2 validation-only screening (>=3 seeds) -> (Stage 3
Optuna tuning of the top 2-3 lives in run_tune_final.py). TEST split is never
touched here.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system, JointSeqDataset
from sim.network import build_network, dc_jacobian
from train.common import set_seed, uq_detection_metrics, localization_metrics
from candidates.egt_net import EGTNet, egt_loss
from candidates.egt_net_sngp import EGTNetSNGP, sngp_loss
from candidates.egt_net_distill import EGTNetStudent, distill_loss
from candidates.pra_net import PRANet, pra_loss


def make_loader(X, y, ybus, bs=128, shuffle=False):
    return DataLoader(JointSeqDataset(X, y, ybus), batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def eval_candidate(detect_fn, loader, device):
    probs, uncs, ys, loc_probs, ybuses = [], [], [], [], []
    for xb, yb, ybusb in loader:
        xb = xb.to(device)
        p, u = detect_fn(xb)
        probs.append(p.cpu().numpy())
        uncs.append(u.cpu().numpy())
        ys.append(yb.numpy())
        ybuses.append(ybusb.numpy())
    probs, uncs, ys = np.concatenate(probs), np.concatenate(uncs), np.concatenate(ys)
    det_m = uq_detection_metrics(ys, uncs)
    det_m["auroc_prob"] = float(np.nan)
    try:
        from sklearn.metrics import roc_auc_score
        det_m["auroc_prob"] = float(roc_auc_score(ys, probs))
    except ValueError:
        pass
    return det_m, probs, uncs, ys


@torch.no_grad()
def eval_localization(model_forward_loc, loader, device):
    loc_probs, ybuses = [], []
    for xb, yb, ybusb in loader:
        xb = xb.to(device)
        lp = model_forward_loc(xb).cpu().numpy()
        loc_probs.append(lp)
        ybuses.append(ybusb.numpy())
    return localization_metrics(np.concatenate(ybuses), np.concatenate(loc_probs))


def build_and_train(cand_name, data, jac, device, epochs, seed, gamma_loc=1.0, subset=None,
                     lr=1e-3, gru_hidden=32, gcn_hidden=32):
    set_seed(seed)
    n_branch, n_bus, branches = jac["n_branch"], jac["n_bus"], jac["branches"]
    Xtr, ytr, ybustr = data["X_train"], data["y_train"], data["ybus_train"]
    if subset:
        idx = np.random.RandomState(seed).choice(len(Xtr), size=min(subset, len(Xtr)), replace=False)
        Xtr, ytr, ybustr = Xtr[idx], ytr[idx], ybustr[idx]
    train_loader = make_loader(Xtr, ytr, ybustr, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], data["ybus_val"])

    if cand_name == "EGT-Net":
        model = EGTNet(n_branch, n_bus, branches, gru_hidden=gru_hidden, gcn_hidden=gcn_hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            for xb, yb, ybusb in train_loader:
                xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                opt.zero_grad()
                loss, _ = egt_loss(model, xb, yb, ybusb, ep, epochs, gamma_loc)
                loss.backward(); opt.step()
        detect_fn = model.detect
        loc_fn = lambda xb: torch.sigmoid(model(xb)[1])

    elif cand_name == "EGT-Net-NoGCN":
        model = EGTNet(n_branch, n_bus, branches, gru_hidden=gru_hidden, gcn_hidden=gcn_hidden, use_gcn=False).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            for xb, yb, ybusb in train_loader:
                xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                opt.zero_grad()
                loss, _ = egt_loss(model, xb, yb, ybusb, ep, epochs, gamma_loc)
                loss.backward(); opt.step()
        detect_fn = model.detect
        loc_fn = lambda xb: torch.sigmoid(model(xb)[1])

    elif cand_name == "EGT-Net-SNGP":
        model = EGTNetSNGP(n_branch, n_bus, branches, gru_hidden=gru_hidden, gcn_hidden=gcn_hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            for xb, yb, ybusb in train_loader:
                xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                opt.zero_grad()
                loss, _ = sngp_loss(model, xb, yb, ybusb, gamma_loc)
                loss.backward(); opt.step()
        model.fit_gp_laplace(train_loader, device)
        detect_fn = model.detect
        loc_fn = lambda xb: torch.sigmoid(model(xb)[1])

    elif cand_name == "EGT-Net-Distill":
        teachers = []
        for j in range(3):
            set_seed(seed * 100 + j)
            t = EGTNet(n_branch, n_bus, branches, use_evidential=False).to(device)
            topt = torch.optim.Adam(t.parameters(), lr=1e-3)
            for ep in range(max(5, epochs // 2)):
                t.train()
                for xb, yb, ybusb in train_loader:
                    xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                    topt.zero_grad()
                    loss, _ = egt_loss(t, xb, yb, ybusb, ep, epochs, gamma_loc)
                    loss.backward(); topt.step()
            teachers.append(t)
        student = EGTNetStudent(n_branch, n_bus, branches).to(device)
        sopt = torch.optim.Adam(student.parameters(), lr=1e-3)
        for ep in range(epochs):
            student.train()
            for xb, yb, ybusb in train_loader:
                xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                sopt.zero_grad()
                loss, _ = distill_loss(student, teachers, xb, yb, ybusb, gamma_loc=gamma_loc)
                loss.backward(); sopt.step()

        def detect_fn(xb):
            det_logits, _, u_pred = student.forward_full(xb)
            return torch.softmax(det_logits, -1)[:, 1], u_pred
        model = student
        loc_fn = lambda xb: torch.sigmoid(student.forward_full(xb)[1])

    elif cand_name == "PRA-Net":
        n_red = jac["H_dc"].shape[1]
        H_dc_t = torch.tensor(jac["H_dc"], dtype=torch.float32, device=device)
        r_inv = torch.ones(jac["H_dc"].shape[0], device=device)  # fixed unit weighting (see pra_net.py docstring)
        model = PRANet(n_branch, n_bus, branches, H_dc_t, r_inv, gru_hidden=gru_hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            for xb, yb, ybusb in train_loader:
                xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                opt.zero_grad()
                loss, _ = pra_loss(model, xb, yb, ybusb, ep, epochs, gamma_loc)
                loss.backward(); opt.step()
        detect_fn = model.detect
        loc_fn = lambda xb: torch.sigmoid(model(xb)[1])
    else:
        raise ValueError(cand_name)

    return model, detect_fn, loc_fn, val_loader


def run_stage1_smoke(system="case30", epochs=3, subset=4000, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    results = {}
    for cand in ["EGT-Net", "EGT-Net-SNGP", "EGT-Net-Distill", "PRA-Net"]:
        t0 = time.time()
        try:
            model, detect_fn, loc_fn, val_loader = build_and_train(
                cand, data, jac, device, epochs, seed, subset=subset)
            det_m, probs, uncs, ys = eval_candidate(detect_fn, val_loader, device)
            loc_m = eval_localization(loc_fn, val_loader, device)
            collapsed = (probs.std() < 1e-4)
            results[cand] = dict(ok=True, collapsed=bool(collapsed), time_sec=time.time() - t0,
                                  det_auroc=det_m.get("auroc_prob"), loc_macro_f1=loc_m["macro_f1"],
                                  params=sum(p.numel() for p in model.parameters()))
            print(f"[smoke] {cand}: auroc={det_m.get('auroc_prob'):.3f} loc_f1={loc_m['macro_f1']:.3f} "
                  f"collapsed={collapsed} time={time.time()-t0:.1f}s params={results[cand]['params']}")
        except Exception as e:
            results[cand] = dict(ok=False, error=str(e))
            print(f"[smoke] {cand}: FAILED - {e}")
    return results


def run_stage2_screening(system="case30", epochs=20, seeds=(0, 1, 2), out_path="results/candidates/stage2_screening.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_results = {}
    for cand in ["EGT-Net", "EGT-Net-NoGCN", "EGT-Net-SNGP", "EGT-Net-Distill", "PRA-Net"]:
        runs = []
        for seed in seeds:
            t0 = time.time()
            model, detect_fn, loc_fn, val_loader = build_and_train(cand, data, jac, device, epochs, seed)
            train_time = time.time() - t0
            det_m, probs, uncs, ys = eval_candidate(detect_fn, val_loader, device)
            loc_m = eval_localization(loc_fn, val_loader, device)
            t1 = time.time()
            with torch.no_grad():
                for xb, _, _ in val_loader:
                    detect_fn(xb.to(device))
            infer_time = (time.time() - t1) / len(data["X_val"])
            runs.append(dict(seed=seed, train_time=train_time, infer_time_per_sample=infer_time,
                              params=sum(p.numel() for p in model.parameters()), **det_m, **{f"loc_{k}": v for k, v in loc_m.items()}))
            print(f"[stage2] {cand} seed={seed}: auroc={det_m['auroc']:.3f} d_error={det_m['d_error']:.3f} "
                  f"loc_macro_f1={loc_m['macro_f1']:.3f} train_time={train_time:.1f}s")
        keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
        agg = {f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in keys}
        agg.update({f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys})
        all_results[cand] = dict(runs=runs, agg=agg)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
    return all_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "screen"], required=True)
    ap.add_argument("--system", default="case30")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "smoke":
        run_stage1_smoke(args.system, epochs=args.epochs or 3)
    else:
        run_stage2_screening(args.system, epochs=args.epochs or 20)
