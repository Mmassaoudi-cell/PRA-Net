"""
Adversarial-robustness diagnostics for PRA-Net, addressing the reviewer
concern that near-zero C&W attack success could reflect gradient masking
rather than genuine robustness (Athalye et al. 2018; Carlini et al. 2019).

Three diagnostics, all against the frozen PRA-Net (FINAL_MODEL_CONFIG.yaml):

1. Gradient-norm trajectory: compares ||d(attack_loss)/d(c_adv)|| across the
   C&W optimization for PRA-Net vs. a plain-softmax source backbone (LSTM).
   Vanishing/near-zero gradients throughout (not just at convergence) are a
   classic masking signature.
2. Adaptive attack on the actual defense signal: the standard C&W attack
   (src/attacks/cw_attack.py) targets flipping the Dirichlet mean-probability
   class prediction via a log(p) proxy. The model's own adversarial-FDIA
   *detection* claim, however, thresholds epistemic uncertainty u = 2/S
   directly (mirroring the source paper's own MI-threshold detector). This
   diagnostic instead directly minimizes u (uncertainty-evasion attack) and
   reports success against a TPR95-calibrated u-threshold -- attacking the
   quantity actually used for the detection claim, not a proxy for it.
3. Transfer (black-box) attack: adversarial examples crafted with full
   white-box access to a *different*, plain-softmax model (FCNN) are
   evaluated against frozen PRA-Net (no gradient access used against PRA-Net
   itself) -- a genuine black-box/gray-box threat-model instantiation.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system
from sim.network import build_network, dc_jacobian
from models.backbones import build_backbone
from candidates.pra_net import PRANet
from attacks.cw_attack import craft_adversarial_fdia, cw_loss
from train.run_final_test import train_final_pranet, PRANetCWWrapper
from train.common import set_seed, uq_detection_metrics


@torch.no_grad()
def u_threshold_at_tpr95(model: PRANet, X_test, y_test, device) -> float:
    """Calibrate the uncertainty-threshold operating point at TPR95 on the
    clean TEST distribution, mirroring the source paper's own MI-threshold
    detector calibration and this project's uq_detection_metrics convention."""
    from sklearn.metrics import roc_curve
    xb = torch.tensor(X_test, dtype=torch.float32, device=device)
    _, u = model.detect(xb)
    u = u.cpu().numpy()
    fpr, tpr, thr = roc_curve(y_test, u)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(thr[idx])


def gradient_norm_trajectory(model_wrapper, Z_batch, buses_batch, H_dc, non_slack, n_branch, n_bus,
                              device, n_iter=500, lr=0.01, C=10.0):
    """Runs a single C-value C&W optimization while recording the gradient
    norm w.r.t. c_adv at every iteration (adapted from cw_attack.craft_adversarial_fdia)."""
    B = Z_batch.shape[0]
    n_red = H_dc.shape[1]
    idx_map = {b: i for i, b in enumerate(non_slack)}
    H_dc_t = torch.tensor(H_dc, dtype=torch.float32, device=device)
    mask = torch.zeros(B, n_red, device=device)
    for i, buses in enumerate(buses_batch):
        for b in buses:
            if b in idx_map:
                mask[i, idx_map[b]] = 1.0
    Z_batch = Z_batch.to(device)
    for p in model_wrapper.parameters():
        p.requires_grad = False

    c_adv = torch.zeros(B, n_red, device=device, requires_grad=True)
    opt = torch.optim.Adam([c_adv], lr=lr)
    grad_norms = []
    with torch.backends.cudnn.flags(enabled=False):
        for it in range(n_iter):
            opt.zero_grad()
            c_masked = c_adv * mask
            a = c_masked @ H_dc_t.T
            Z_adv = Z_batch.clone()
            Z_adv[:, -1, :n_branch] = Z_adv[:, -1, :n_branch] + a[:, :n_branch]
            Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] = (
                Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] + a[:, n_branch:])
            logits = model_wrapper(Z_adv)
            margin = cw_loss(logits, target_class=0)
            l2 = (a ** 2).sum(-1)
            loss = (l2 + C * margin).sum()
            loss.backward()
            grad_norms.append(c_adv.grad.norm(dim=-1).mean().item())
            opt.step()
    return np.array(grad_norms)


def uncertainty_evasion_attack(model: PRANet, Z_batch, buses_batch, H_dc, non_slack, n_branch, n_bus,
                                device, u_thresh, n_iter=500, lr=0.01, C_grid=(1.0, 10.0, 100.0)):
    """Adaptive attack directly minimizing epistemic uncertainty u (the
    model's own claimed detection signal), rather than the log(p) proxy used
    by the standard C&W attack. Success = u(Z_adv) < u_thresh (evades the
    TPR95-calibrated detection threshold)."""
    B = Z_batch.shape[0]
    n_red = H_dc.shape[1]
    idx_map = {b: i for i, b in enumerate(non_slack)}
    H_dc_t = torch.tensor(H_dc, dtype=torch.float32, device=device)
    mask = torch.zeros(B, n_red, device=device)
    for i, buses in enumerate(buses_batch):
        for b in buses:
            if b in idx_map:
                mask[i, idx_map[b]] = 1.0
    Z_batch = Z_batch.to(device)
    for p in model.parameters():
        p.requires_grad = False

    best_Z = Z_batch.clone()
    best_success = torch.zeros(B, dtype=torch.bool, device=device)
    best_norm = torch.full((B,), float("inf"), device=device)

    with torch.backends.cudnn.flags(enabled=False):
        for C in C_grid:
            c_adv = torch.zeros(B, n_red, device=device, requires_grad=True)
            opt = torch.optim.Adam([c_adv], lr=lr)
            for it in range(n_iter):
                opt.zero_grad()
                c_masked = c_adv * mask
                a = c_masked @ H_dc_t.T
                Z_adv = Z_batch.clone()
                Z_adv[:, -1, :n_branch] = Z_adv[:, -1, :n_branch] + a[:, :n_branch]
                Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] = (
                    Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] + a[:, n_branch:])
                _, u = model.detect(Z_adv)
                l2 = (a ** 2).sum(-1)
                loss = (l2 + C * u).sum()
                loss.backward()
                opt.step()
            with torch.no_grad():
                c_masked = c_adv * mask
                a = c_masked @ H_dc_t.T
                Z_adv = Z_batch.clone()
                Z_adv[:, -1, :n_branch] += a[:, :n_branch]
                Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] += a[:, n_branch:]
                _, u = model.detect(Z_adv)
                success = u < u_thresh
                norm = a.norm(dim=-1)
                improve = success & (norm < best_norm)
                best_Z[improve] = Z_adv[improve]
                best_success = best_success | success
                best_norm = torch.where(improve, norm, best_norm)
    return best_Z, best_success.cpu().numpy()


def run(system="case30", epochs=25, seed=0, n_iter=500, out_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    n_branch, n_bus, non_slack = jac["n_branch"], jac["n_bus"], jac["non_slack"]
    with open("FINAL_MODEL_CONFIG.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"[{system}] training frozen PRA-Net (seed={seed}) for diagnostics...")
    pranet = train_final_pranet(data, jac, device, epochs, seed, cfg)
    pranet.eval()

    d = data["meta"]["d"]
    lstm = build_backbone("LSTM", d, seq_len=6).to(device)
    ckpt_path = f"models_ckpt/{system}/LSTM_basic.pt"
    lstm.load_state_dict(torch.load(ckpt_path, map_location=device))
    lstm.eval()
    fcnn = build_backbone("FCNN", d, seq_len=6).to(device)
    fcnn.load_state_dict(torch.load(f"models_ckpt/{system}/FCNN_basic.pt", map_location=device))
    fcnn.eval()

    case_names = [k[5:-2] for k in data if k.startswith("case_") and k.endswith("_X")]
    results = {"system": system, "seed": seed, "cases": {}}

    u_thresh = u_threshold_at_tpr95(pranet, data["X_test"], data["y_test"], device)
    results["u_threshold_tpr95"] = u_thresh
    print(f"[{system}] u-threshold @ TPR95 (clean TEST) = {u_thresh:.6g}")

    for case in case_names:
        Xc = data[f"case_{case}_X"]
        buses_c = data[f"case_{case}_buses"].tolist()
        Xc_t = torch.tensor(Xc, dtype=torch.float32)
        buses_batch = [buses_c] * len(Xc)

        # --- 1. Gradient-norm trajectory: PRA-Net vs LSTM (both, same recipe) ---
        gn_pranet = gradient_norm_trajectory(PRANetCWWrapper(pranet), Xc_t[:64], buses_batch[:64],
                                              jac["H_dc"], non_slack, n_branch, n_bus, device, n_iter=n_iter)
        gn_lstm = gradient_norm_trajectory(lstm, Xc_t[:64], buses_batch[:64],
                                            jac["H_dc"], non_slack, n_branch, n_bus, device, n_iter=n_iter)

        # --- 2. Adaptive uncertainty-evasion attack (targets u directly) ---
        Z_adv_u, success_u = uncertainty_evasion_attack(
            pranet, Xc_t, buses_batch, jac["H_dc"], non_slack, n_branch, n_bus, device,
            u_thresh, n_iter=n_iter)

        # --- 3. Transfer/black-box attack: craft on FCNN, test on PRA-Net ---
        bs = 64
        Z_adv_transfer_list, success_transfer_list = [], []
        for i in range(0, len(Xc), bs):
            xb = Xc_t[i:i + bs]
            z_adv, success_wb, _, _ = craft_adversarial_fdia(
                fcnn, xb, buses_batch[i:i + bs], jac["H_dc"], non_slack, n_branch, n_bus,
                device, C_grid=(1.0, 10.0, 100.0), n_iter=n_iter, lr=0.01)
            Z_adv_transfer_list.append(z_adv.cpu().numpy())
            success_transfer_list.append(success_wb.cpu().numpy())
        Z_adv_transfer = np.concatenate(Z_adv_transfer_list)
        success_transfer_wb = np.concatenate(success_transfer_list)  # fooled FCNN itself
        with torch.no_grad():
            zb = torch.tensor(Z_adv_transfer, dtype=torch.float32, device=device)
            p_pranet, u_pranet = pranet.detect(zb)
            pred_pranet = (p_pranet >= 0.5).long().cpu().numpy()
        transfer_success_on_pranet = (pred_pranet == 0) & success_transfer_wb  # fools FCNN AND transfers
        transfer_evades_u = (u_pranet.cpu().numpy() < u_thresh) & success_transfer_wb

        results["cases"][case] = dict(
            n=len(Xc),
            gradient_norm_pranet_mean=float(gn_pranet.mean()),
            gradient_norm_pranet_median=float(np.median(gn_pranet)),
            gradient_norm_pranet_final10pct_mean=float(gn_pranet[-max(1, n_iter // 10):].mean()),
            gradient_norm_lstm_mean=float(gn_lstm.mean()),
            gradient_norm_lstm_median=float(np.median(gn_lstm)),
            gradient_norm_lstm_final10pct_mean=float(gn_lstm[-max(1, n_iter // 10):].mean()),
            uncertainty_evasion_success_rate=float(success_u.mean()),
            uncertainty_evasion_n_success=int(success_u.sum()),
            transfer_fools_fcnn_rate=float(success_transfer_wb.mean()),
            transfer_fools_pranet_classlabel_rate=float(transfer_success_on_pranet.mean()),
            transfer_evades_pranet_uncertainty_rate=float(transfer_evades_u.mean()),
        )
        print(f"[{system}] case {case}: grad_norm(PRA-Net)={gn_pranet.mean():.3e} "
              f"grad_norm(LSTM)={gn_lstm.mean():.3e} | u-evasion success={success_u.mean():.2%} "
              f"| transfer fools FCNN={success_transfer_wb.mean():.2%} "
              f"transfers-to-PRA-Net(label)={transfer_success_on_pranet.mean():.2%} "
              f"transfers-to-PRA-Net(u-evasion)={transfer_evades_u.mean():.2%}")

    out_path = out_path or f"results/robustness/{system}_adversarial_diagnostics.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Wrote {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="case30")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_iter", type=int, default=500)
    args = ap.parse_args()
    run(args.system, args.epochs, args.seed, args.n_iter)
