"""
Robustness/generalization experiments (master task Sec. 20-21) for the
final, frozen model (PRA-Net, FINAL_MODEL_CONFIG.yaml): measurement-noise
sweep, missing-channel sweep, and training-time label-noise sweep -- run
post-freeze on the TEST split (this is final reporting, not candidate
re-selection; FINAL_MODEL_CONFIG.yaml is never modified based on these
results).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system
from sim.network import build_network, dc_jacobian
from train.common import set_seed, binary_metrics
from train.run_candidates import make_loader, eval_candidate
from candidates.pra_net import PRANet, pra_loss


def train_final(data, jac, cfg, device, epochs, seed, label_noise=0.0):
    set_seed(seed)
    n_branch, n_bus, branches = jac["n_branch"], jac["n_bus"], jac["branches"]
    hp = cfg["hyperparameters"]
    ytr = data["y_train"].copy()
    if label_noise > 0:
        rng = np.random.RandomState(seed)
        flip = rng.rand(len(ytr)) < label_noise
        ytr[flip] = 1 - ytr[flip]
    train_loader = make_loader(data["X_train"], ytr, data["ybus_train"], shuffle=True)
    H_dc_t = torch.tensor(jac["H_dc"], dtype=torch.float32, device=device)
    r_inv = torch.ones(jac["H_dc"].shape[0], device=device)
    model = PRANet(n_branch, n_bus, branches, H_dc_t, r_inv, gru_hidden=hp["gru_hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    for ep in range(epochs):
        model.train()
        for xb, yb, ybusb in train_loader:
            xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
            opt.zero_grad()
            loss, _ = pra_loss(model, xb, yb, ybusb, ep, epochs, gamma_loc=hp["gamma_loc"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def eval_probs(model, X, y, ybus, device):
    loader = make_loader(X, y, ybus)
    _, probs, uncs, ys = eval_candidate(model.detect, loader, device)
    return binary_metrics(ys, probs)


def noise_sweep(model, data, device, factors=(0.0, 0.5, 1.0, 2.0, 4.0)):
    out = {}
    Xte, yte, ybus_te = data["X_test"], data["y_test"], data["ybus_test"]
    base_std = data["X_train"].std()
    for f in factors:
        Xn = Xte + np.random.RandomState(0).normal(0, f * base_std, size=Xte.shape).astype(np.float32)
        m = eval_probs(model, Xn, yte, ybus_te, device)
        out[f"noise_x{f}"] = m
        print(f"[robust] noise factor {f}: macro_f1={m['macro_f1']:.4f} auroc={m['roc_auc']:.4f}")
    return out


def missing_channel_sweep(model, data, device, fracs=(0.0, 0.1, 0.2, 0.4)):
    out = {}
    Xte, yte, ybus_te = data["X_test"], data["y_test"], data["ybus_test"]
    d = Xte.shape[-1]
    for frac in fracs:
        rng = np.random.RandomState(0)
        mask = rng.rand(d) >= frac
        Xm = Xte * mask[None, None, :]
        m = eval_probs(model, Xm, yte, ybus_te, device)
        out[f"missing_{frac}"] = m
        print(f"[robust] missing frac {frac}: macro_f1={m['macro_f1']:.4f} auroc={m['roc_auc']:.4f}")
    return out


def label_noise_sweep(data, jac, cfg, device, epochs, seed, levels=(0.0, 0.05, 0.1, 0.2)):
    out = {}
    for lvl in levels:
        model = train_final(data, jac, cfg, device, epochs, seed, label_noise=lvl)
        m = eval_probs(model, data["X_test"], data["y_test"], data["ybus_test"], device)
        out[f"label_noise_{lvl}"] = m
        print(f"[robust] label noise {lvl}: macro_f1={m['macro_f1']:.4f} auroc={m['roc_auc']:.4f}")
    return out


def run_all(system="case30", epochs=25, seed=0, out_path="results/robustness/robustness_results.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    with open("FINAL_MODEL_CONFIG.yaml") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model = train_final(data, jac, cfg, device, epochs, seed)
    results = dict(
        noise=noise_sweep(model, data, device),
        missing_channels=missing_channel_sweep(model, data, device),
        label_noise=label_noise_sweep(data, jac, cfg, device, epochs, seed),
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="case30")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_all(args.system, args.epochs, args.seed)
