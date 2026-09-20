"""
SOURCE_METHOD_REPRODUCTION, localization half (Sec. IV-B / Table V-VI /
Figs. 6-8): VAE-guided counterfactual search + MI-gradient attribution +
topology neighborhood-consistency filter. Run once per system on the
ResNet backbone (the source paper's own illustrative choice for Figs. 4, 6
and 7), reusing the basic-detector checkpoint saved by run_reproduction.py.
A single VAE is trained per system (shared across backbones, since it only
needs to model the measurement-sequence manifold, not any one detector).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system, SeqDataset
from sim.network import build_network, dc_jacobian, adjacency_graph
from models.backbones import build_backbone
from models.uncertainty import MultiHeadEnsemble, multihead_loss, LLLAMultiHeadEnsemble
from models.vae_localization import (SeqVAE, vae_loss, counterfactual_search,
                                      channel_contribution_gradients, localization_scores,
                                      neighborhood_consistency_filter, calibrate_localization_threshold)
from train.common import set_seed, localization_metrics
from attacks.cw_attack import craft_adversarial_fdia


def train_vae(X_train, device, epochs=30, latent_dim=32):
    ds = SeqDataset(X_train, np.zeros(len(X_train), dtype=np.int64))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    k, d = X_train.shape[1], X_train.shape[2]
    vae = SeqVAE(k, d, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for ep in range(epochs):
        for xb, _ in loader:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = vae(xb)
            loss, recon, kl = vae_loss(xb, x_hat, mu, logvar)
            loss.backward()
            opt.step()
    return vae


def build_llla_ensemble(model, train_loader, device, m=5, hidden=64):
    ens = MultiHeadEnsemble(model, m=m, hidden=hidden).to(device)
    opt = torch.optim.Adam(ens.heads.parameters(), lr=1e-3)
    for ep in range(15):
        ens.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss, _ = multihead_loss(ens(xb), yb)
            loss.backward()
            opt.step()
    llla = LLLAMultiHeadEnsemble(ens, prior_precision=5.0)
    llla.fit(train_loader, device)
    return llla


def run(system: str, backbone: str = "ResNet", seed: int = 0, epochs_vae: int = 30,
        out_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    graph = adjacency_graph(net)
    d = data["meta"]["d"]
    n_branch, n_bus = jac["n_branch"], jac["n_bus"]

    model = build_backbone(backbone, d, seq_len=6).to(device)
    ckpt_path = f"models_ckpt/{system}/{backbone}_basic.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    train_loader = DataLoader(SeqDataset(data["X_train"], data["y_train"]), batch_size=128, shuffle=True)
    print(f"[{system}] training shared VAE...")
    vae = train_vae(data["X_train"], device, epochs=epochs_vae)
    print(f"[{system}] fitting LLLA multihead ensemble on {backbone}...")
    llla = build_llla_ensemble(model, train_loader, device)

    # Calibrate localization threshold from 1000 historical non-attacked test sequences
    neg_idx = np.where(data["y_test"] == 0)[0]
    n_calib = min(1000, len(neg_idx))
    calib_idx = np.random.RandomState(seed).choice(neg_idx, size=n_calib, replace=False)
    Z_calib = torch.tensor(data["X_test"][calib_idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        pass  # scores below use gradients, computed per-batch
    calib_scores = []
    bs = 64
    for i in range(0, len(Z_calib), bs):
        zb = Z_calib[i:i + bs]
        G = channel_contribution_gradients(llla, zb, mc_samples=20)
        dZ = torch.zeros_like(zb)  # no counterfactual shift for calibration on non-attacked data
        s = localization_scores(G, dZ.abs() + 1e-6, n_branch, n_bus, jac["branches"])
        calib_scores.append(s)
    calib_scores = np.concatenate(calib_scores)
    S_thresh = calibrate_localization_threshold(calib_scores)
    print(f"[{system}] localization threshold S={S_thresh:.4f}")

    def topk_hit_rate(scores, ybus, k):
        order = np.argsort(-scores, axis=1)[:, :k]
        hits = [ybus[i, order[i]].sum() > 0 for i in range(len(scores)) if ybus[i].sum() > 0]
        return float(np.mean(hits)) if hits else float("nan")

    case_names = [k[5:-2] for k in data if k.startswith("case_") and k.endswith("_X")]
    results = {"system": system, "backbone": backbone, "S_thresh": float(S_thresh), "cases": {}}
    for case in case_names:
        Xc_base = torch.tensor(data[f"case_{case}_X"], dtype=torch.float32, device=device)
        buses_c = data[f"case_{case}_buses"].tolist()
        # Localization (Sec. IV-B) is only meaningful for sequences that are
        # *genuinely* successful adversarial FDIAs (Z^adv that fooled the
        # detector) -- craft these with the same C&W attack used for
        # detection (run_reproduction.py), then evaluate localization only
        # on the subset where the attack actually succeeded.
        H_dc_np = jac["H_dc"]
        Xc_all, success_all = [], []
        for i in range(0, len(Xc_base), bs):
            zb = Xc_base[i:i + bs]
            z_adv, success, _, _ = craft_adversarial_fdia(
                model, zb, [buses_c] * len(zb), H_dc_np, jac["non_slack"], n_branch, n_bus,
                device, C_grid=(1.0, 10.0), n_iter=150, lr=0.01)
            Xc_all.append(z_adv)
            success_all.append(success)
        Xc_all = torch.cat(Xc_all)
        success_all = torch.cat(success_all).cpu().numpy()
        Xc = Xc_all[success_all]
        ybus_c = data[f"case_{case}_ybus"][success_all]
        print(f"[{system}] case {case}: {success_all.sum()}/{len(success_all)} sequences are "
              f"genuinely successful C&W adversarial FDIAs (localization evaluated on these only)")
        if len(Xc) == 0:
            results["cases"][case] = dict(note="0 successful adversarial examples at this C&W budget")
            continue
        all_scores, all_scores_raw = [], []
        for i in range(0, len(Xc), bs):
            zb = Xc[i:i + bs]
            Z_minus = counterfactual_search(vae, llla, zb, mc_samples=15)
            G = channel_contribution_gradients(llla, zb, mc_samples=20)
            dZ = Z_minus - zb
            s = localization_scores(G, dZ, n_branch, n_bus, jac["branches"])
            s_filtered = np.stack([neighborhood_consistency_filter(row, graph) for row in s])
            all_scores.append(s_filtered)
            all_scores_raw.append(s)
        all_scores = np.concatenate(all_scores)
        all_scores_raw = np.concatenate(all_scores_raw)
        top1 = topk_hit_rate(all_scores_raw, ybus_c, 1)
        top3 = topk_hit_rate(all_scores_raw, ybus_c, 3)
        top5 = topk_hit_rate(all_scores_raw, ybus_c, 5)
        print(f"[{system}] case {case}: threshold-free top-1/3/5 hit-rate = "
              f"{top1:.3f}/{top3:.3f}/{top5:.3f} (chance top-5 of {n_bus} ~= {5/n_bus:.3f})")
        y_pred = (all_scores > S_thresh).astype(int)

        tp = ((y_pred == 1) & (ybus_c == 1)).sum()
        fp = ((y_pred == 1) & (ybus_c == 0)).sum()
        fn = ((y_pred == 0) & (ybus_c == 1)).sum()
        tn = ((y_pred == 0) & (ybus_c == 0)).sum()
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        macro_f1_metrics = localization_metrics(ybus_c, all_scores / (all_scores.max() + 1e-9))

        results["cases"][case] = dict(
            n=len(Xc), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
            recall=float(recall), specificity=float(specificity),
            macro_f1=macro_f1_metrics["macro_f1"], macro_pr_auc=macro_f1_metrics["macro_pr_auc"],
            top1_hit_rate=top1, top3_hit_rate=top3, top5_hit_rate=top5,
            chance_top5_rate=5 / n_bus)
        print(f"[{system}] case {case}: recall={recall:.3f} specificity={specificity:.3f} "
              f"macro_f1={macro_f1_metrics['macro_f1']:.3f}")

    out_path = out_path or f"results/reproduction/{system}_localization_reproduction.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="case30")
    ap.add_argument("--backbone", default="ResNet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.system, args.backbone, args.seed)
