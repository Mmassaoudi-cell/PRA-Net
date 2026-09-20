"""
Ablation study (master task Sec. 19) for the final selected model, PRA-Net
(Physics-Residual Attention Network, FINAL_MODEL_CONFIG.yaml): Full model
vs. Full-minus-component, isolating each of PRA-Net's mechanisms
(MODEL_CANDIDATES.md, Candidate 4):
  -Physics    : the DC-WLS residual auxiliary feature removed (raw
                measurements only feed the per-bus GRU)
  -AttnBias   : attention kept, but the residual-magnitude bias term is
                disabled (eta forced to 0) -- isolates "attention learns
                freely" from "attention is nudged by the physics prior"
  -Attention  : self-attention over bus tokens replaced entirely by a GCN
                (directly comparable to Candidate 1's topology mechanism,
                isolating "attention vs. graph-conv" as the localization
                mechanism)
  -JointLoc   : localization head present but receives no training signal
                (gamma_loc=0) -- the closest same-architecture analogue of
                the source paper's post-hoc (non-jointly-trained)
                gradient-attribution approach.
  Plain       : both the physics-residual feature and the attention
                mechanism removed simultaneously (per-bus GRU + mean-pool +
                heads only, joint supervision retained) -- the most
                conservative control, isolating how much of PRA-Net's gain
                is attributable to jointly-supervised localization alone
                versus the physics-attention architecture specifically.
Run with >=5 seeds on the frozen model's TEST split (per FINAL_MODEL_CONFIG.yaml;
this is final reporting, not re-selection -- the config is not modified
based on these results). A two-sample Welch's t-test against the Full model
is computed for every variant to support causal claims with more than a
mean/std comparison.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system
from sim.network import build_network, dc_jacobian
from train.common import set_seed
from train.run_candidates import make_loader, eval_candidate, eval_localization
from candidates.pra_net import PRANet, pra_loss
from candidates.egt_net import EGTNet, egt_loss


VARIANTS = ["Full", "-Physics", "-AttnBias", "-Attention", "-JointLoc", "Plain"]


def train_variant(variant, data, jac, cfg, device, epochs, seed, split="test"):
    set_seed(seed)
    n_branch, n_bus, branches = jac["n_branch"], jac["n_bus"], jac["branches"]
    hp = cfg["hyperparameters"]
    gamma_loc = 0.0 if variant == "-JointLoc" else hp["gamma_loc"]
    train_loader = make_loader(data["X_train"], data["y_train"], data["ybus_train"], shuffle=True)
    eval_loader = make_loader(data[f"X_{split}"], data[f"y_{split}"], data[f"ybus_{split}"])

    if variant == "-Attention":
        # isolate "attention vs. graph-conv" by swapping in Candidate 1's
        # GCN-based topology mechanism on the same GRU backbone/heads family
        model = EGTNet(n_branch, n_bus, branches, gru_hidden=hp["gru_hidden"],
                        gcn_hidden=hp["gru_hidden"], use_gcn=True, use_evidential=True).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        for ep in range(epochs):
            model.train()
            for xb, yb, ybusb in train_loader:
                xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                opt.zero_grad()
                loss, _ = egt_loss(model, xb, yb, ybusb, ep, epochs, gamma_loc=gamma_loc)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        detect_fn, loc_fn = model.detect, lambda xb: torch.sigmoid(model(xb)[1])
    else:
        use_physics = variant not in ("-Physics", "Plain")
        use_attention = variant != "Plain"
        H_dc_t = torch.tensor(jac["H_dc"], dtype=torch.float32, device=device)
        r_inv = torch.ones(jac["H_dc"].shape[0], device=device)
        model = PRANet(n_branch, n_bus, branches, H_dc_t, r_inv, gru_hidden=hp["gru_hidden"],
                        use_physics=use_physics, use_attention=use_attention).to(device)
        if variant == "-AttnBias":
            model.eta.requires_grad_(False)
            model.eta.data.zero_()
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        for ep in range(epochs):
            model.train()
            for xb, yb, ybusb in train_loader:
                xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
                opt.zero_grad()
                loss, _ = pra_loss(model, xb, yb, ybusb, ep, epochs, gamma_loc=gamma_loc)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        detect_fn, loc_fn = model.detect, lambda xb: torch.sigmoid(model(xb)[1])

    det_m, probs, uncs, ys = eval_candidate(detect_fn, eval_loader, device)
    loc_m = eval_localization(loc_fn, eval_loader, device)
    from train.common import binary_metrics
    clean_m = binary_metrics(ys, probs)
    return clean_m, loc_m


def run_ablation(system="case30", epochs=25, seeds=(0, 1, 2, 3, 4), split="test",
                  out_path="results/ablation/ablation_results.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    with open("FINAL_MODEL_CONFIG.yaml") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = {}
    for variant in VARIANTS:
        runs = []
        for seed in seeds:
            clean_m, loc_m = train_variant(variant, data, jac, cfg, device, epochs, seed, split)
            runs.append(dict(seed=seed, **clean_m, **{f"loc_{k}": v for k, v in loc_m.items()}))
            print(f"[ablation] {variant} seed={seed}: macro_f1={clean_m['macro_f1']:.4f} "
                  f"loc_macro_f1={loc_m['macro_f1']:.4f}")
        keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
        agg = {f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in keys}
        agg.update({f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys})
        results[variant] = dict(runs=runs, agg=agg)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)

    # Welch's t-test of every variant against Full, on both headline metrics
    full_runs = results["Full"]["runs"]
    for variant in VARIANTS:
        if variant == "Full":
            continue
        sig = {}
        for metric in ["macro_f1", "loc_macro_f1"]:
            full_vals = np.array([r[metric] for r in full_runs])
            var_vals = np.array([r[metric] for r in results[variant]["runs"]])
            if np.allclose(full_vals, var_vals):
                t_stat, p_value = float("nan"), 1.0
            else:
                t_stat, p_value = stats.ttest_ind(full_vals, var_vals, equal_var=False)
            sig[metric] = dict(t_stat=float(t_stat), p_value=float(p_value),
                                significant_at_alpha05=bool(p_value < 0.05))
        results[variant]["significance_vs_full"] = sig
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="case30")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    run_ablation(args.system, args.epochs, tuple(args.seeds), args.split)
