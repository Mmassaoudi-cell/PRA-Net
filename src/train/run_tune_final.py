"""
Stage 3 (master task Sec. 9): Optuna (TPE) validation-only hyperparameter
tuning of the top 2-3 candidates surviving Stage 2 screening, followed by
writing FINAL_MODEL_CONFIG.yaml for the single selected model. TEST split is
never touched here.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import optuna
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_system
from sim.network import build_network, dc_jacobian
from train.run_candidates import build_and_train, eval_candidate, eval_localization


def objective_factory(cand_name, data, jac, device, epochs, seeds):
    def objective(trial: optuna.Trial):
        lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 2e-3])
        gru_hidden = trial.suggest_categorical("gru_hidden", [16, 32, 64])
        gcn_hidden = trial.suggest_categorical("gcn_hidden", [16, 32, 64])
        gamma_loc = trial.suggest_categorical("gamma_loc", [0.5, 1.0, 2.0])
        scores = []
        for seed in seeds:
            model, detect_fn, loc_fn, val_loader = build_and_train(
                cand_name, data, jac, device, epochs, seed,
                gamma_loc=gamma_loc, lr=lr, gru_hidden=gru_hidden, gcn_hidden=gcn_hidden)
            det_m, probs, uncs, ys = eval_candidate(detect_fn, val_loader, device)
            loc_m = eval_localization(loc_fn, val_loader, device)
            score = 0.5 * (det_m["auroc_prob"] if not np.isnan(det_m["auroc_prob"]) else 0.5) \
                + 0.5 * loc_m["macro_f1"]
            scores.append(score)
        return float(np.mean(scores))
    return objective


def tune_candidate(cand_name, system="case30", epochs=15, n_trials=15, seeds=(0, 1)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_system(system)
    net = build_network(system)
    jac = dc_jacobian(net)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective_factory(cand_name, data, jac, device, epochs, seeds), n_trials=n_trials)
    print(f"[tune] {cand_name} best value={study.best_value:.4f} params={study.best_params}")
    return study


def write_final_config(cand_name, best_params, system, out_path="FINAL_MODEL_CONFIG.yaml"):
    cfg = dict(
        selected_architecture=cand_name,
        system=system,
        datasets=dict(case30="data/processed/case30/dataset.npz",
                      case118="data/processed/case118/dataset.npz"),
        preprocessing="per-channel z-score standardization fit on TRAIN split only (utils/data.py Standardizer)",
        split="scenario/chronic-level 70/15/15 (DATA_SPLIT_MANIFEST.csv), no random-row leakage",
        hyperparameters=best_params,
        optimizer="Adam",
        loss="evidential Dirichlet detection loss + focal BCE multi-label localization loss (candidates/egt_net.py)",
        model_selection_metric="0.5*val_detection_AUROC + 0.5*val_localization_macro_F1",
        epochs=25,
        early_stopping="best-val-checkpoint over fixed epoch budget",
        seeds_for_final_test="0,1,2,3,4 (>=5 per master-task Sec. 22)",
        class_balancing="focal loss (gamma=2, alpha=0.75) for extreme per-bus imbalance; no oversampling",
        notes="Frozen after Stage 3 validation-only tuning; TEST split not consulted before this point.",
    )
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--system", default="case30")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_trials", type=int, default=15)
    ap.add_argument("--write_final", action="store_true")
    args = ap.parse_args()
    study = tune_candidate(args.candidate, args.system, args.epochs, args.n_trials)
    if args.write_final:
        write_final_config(args.candidate, study.best_params, args.system)
