"""
Final, frozen-model TEST-set evaluation (master task Sec. 15/Step 15): trains
PRA-Net per FINAL_MODEL_CONFIG.yaml across >=5 seeds and reports:
  (a) plain BDD-passing-FDIA detection on TEST (macro-F1, balanced acc,
      minority recall, PR-AUC, ROC-AUC, FPR, FNR) -- comparable to the
      benchmark suite (run_benchmarks.py) and to the source-method backbones'
      "clean" numbers in run_reproduction.py;
  (b) genuine adversarial-FDIA detection via epistemic uncertainty, crafted
      with the SAME C&W attack used throughout this project, against the
      four canonical test cases -- directly comparable to
      SOURCE_METHOD_REPRODUCTION's Table-IV-style uq_report;
  (c) multi-label bus localization on TEST (macro-F1, PR-AUC, precision,
      recall) on the canonical test cases' successful adversarial examples;
  (d) efficiency (params, train time, inference latency/sample).

This file is never re-run to "fix" a disappointing result; per master-task
Sec. 11 the architecture is frozen in FINAL_MODEL_CONFIG.yaml before this
script is ever invoked.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system
from sim.network import build_network, dc_jacobian
from attacks.cw_attack import craft_adversarial_fdia
from train.common import set_seed, uq_detection_metrics, localization_metrics, binary_metrics
from train.run_candidates import make_loader, eval_candidate, eval_localization
from candidates.pra_net import PRANet, pra_loss
from models.evidential import dirichlet_predict
import torch.nn as nn


class PRANetCWWrapper(nn.Module):
    """Adapts PRANet (which returns a (detection, localization) tuple, with
    detection given as Dirichlet evidence rather than raw softmax logits) to
    the plain logits-in/logits-out interface craft_adversarial_fdia expects
    (built originally for the source paper's single-output backbones).
    log(p) stands in for logits: for a 2-class softmax-margin loss, ranking
    by log-probability is equivalent to ranking by probability, and the
    margin log(p_1)-log(p_0) is minimized by exactly the same optimum as
    p_1 - p_0 would be."""
    def __init__(self, pra_net: PRANet):
        super().__init__()
        self.pra_net = pra_net

    def forward(self, Z):
        det_out, _ = self.pra_net(Z)
        p, _ = dirichlet_predict(det_out)
        return p.clamp_min(1e-8).log()


def train_final_pranet(data, jac, device, epochs, seed, cfg):
    set_seed(seed)
    n_branch, n_bus, branches = jac["n_branch"], jac["n_bus"], jac["branches"]
    H_dc_t = torch.tensor(jac["H_dc"], dtype=torch.float32, device=device)
    r_inv = torch.ones(jac["H_dc"].shape[0], device=device)
    hp = cfg["hyperparameters"]
    model = PRANet(n_branch, n_bus, branches, H_dc_t, r_inv, gru_hidden=hp["gru_hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    train_loader = make_loader(data["X_train"], data["y_train"], data["ybus_train"], shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb, ybusb in train_loader:
            xb, yb, ybusb = xb.to(device), yb.to(device), ybusb.to(device)
            opt.zero_grad()
            loss, _ = pra_loss(model, xb, yb, ybusb, ep, epochs, gamma_loc=hp["gamma_loc"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
    return model


def craft_cw_for_case(model, Xc, buses_c, jac, device, n_iter=500):
    n_branch, n_bus = jac["n_branch"], jac["n_bus"]
    bs = 64
    wrapper = PRANetCWWrapper(model)
    all_adv, all_success = [], []
    for i in range(0, len(Xc), bs):
        xb = torch.tensor(Xc[i:i + bs], dtype=torch.float32)
        z_adv, success, _, _ = craft_adversarial_fdia(
            wrapper, xb, [buses_c] * len(xb), jac["H_dc"], jac["non_slack"], n_branch, n_bus,
            device, C_grid=(1.0, 10.0, 100.0), n_iter=n_iter, lr=0.01)
        all_adv.append(z_adv.cpu().numpy())
        all_success.append(success.cpu().numpy())
    return np.concatenate(all_adv), np.concatenate(all_success)


def run_system(system: str, cfg: dict, seeds=(0, 1, 2, 3, 4), epochs=25, out_dir="results/benchmarks"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    case_names = [k[5:-2] for k in data if k.startswith("case_") and k.endswith("_X")]

    per_seed = []
    for seed in seeds:
        t0 = time.time()
        model = train_final_pranet(data, jac, device, epochs, seed, cfg)
        train_time = time.time() - t0
        params = sum(p.numel() for p in model.parameters())

        test_loader = make_loader(data["X_test"], data["y_test"], data["ybus_test"])
        _, probs, uncs, ys = eval_candidate(model.detect, test_loader, device)
        det_m = binary_metrics(ys, probs)  # standard classification metrics, comparable to run_benchmarks.py
        loc_m = eval_localization(lambda xb: torch.sigmoid(model(xb)[1]), test_loader, device)

        t1 = time.time()
        with torch.no_grad():
            for xb, _, _ in test_loader:
                model.detect(xb.to(device))
        infer_time = (time.time() - t1) / len(data["X_test"])

        # Genuine adversarial-FDIA detection (Table-IV-style comparison)
        adv_case_results = {}
        for case in case_names:
            Xc = data[f"case_{case}_X"]
            buses_c = data[f"case_{case}_buses"].tolist()
            Z_adv, success = craft_cw_for_case(model, Xc, buses_c, jac, device)
            Z_adv_succ = Z_adv[success]
            if len(Z_adv_succ) == 0:
                adv_case_results[case] = dict(n_success=0)
                continue
            with torch.no_grad():
                zb = torch.tensor(Z_adv_succ, dtype=torch.float32, device=device)
                _, unc_adv = model.detect(zb)
            mi_clean = uncs  # clean test-set epistemic uncertainty (already computed above)
            labels = np.concatenate([np.zeros(len(mi_clean)), np.ones(len(unc_adv))])
            scores = np.concatenate([mi_clean, unc_adv.cpu().numpy()])
            adv_det_m = uq_detection_metrics(labels, scores)
            adv_loc_m = None
            ybus_succ = data[f"case_{case}_ybus"][success]
            with torch.no_grad():
                zb = torch.tensor(Z_adv_succ, dtype=torch.float32, device=device)
                loc_probs = torch.sigmoid(model(zb)[1]).cpu().numpy()
            adv_loc_m = localization_metrics(ybus_succ, loc_probs)
            adv_case_results[case] = dict(n_success=int(success.sum()), n_total=len(Xc),
                                           detection=adv_det_m, localization=adv_loc_m)

        per_seed.append(dict(seed=seed, train_time_sec=train_time, params=params,
                              infer_time_per_sample=infer_time, clean_detection=det_m,
                              clean_localization=loc_m, adversarial=adv_case_results))
        print(f"[{system}] seed={seed}: clean_macro_f1={det_m['macro_f1']:.4f} "
              f"loc_macro_f1={loc_m['macro_f1']:.4f} train_time={train_time:.1f}s")

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{system}_final_pranet_test.json"
    with open(out_path, "w") as f:
        json.dump(per_seed, f, indent=2, default=float)
    print(f"Wrote {out_path}")
    return per_seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="case30")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    with open("FINAL_MODEL_CONFIG.yaml") as f:
        cfg = yaml.safe_load(f)
    run_system(args.system, cfg, tuple(args.seeds), args.epochs)
