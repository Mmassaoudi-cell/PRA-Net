"""
Aggregates PRA-Net's genuine-adversarial-example detection/localization
metrics across all 5 seeds (rather than reporting a single, potentially
unrepresentative seed), per case. Addresses the requirement to disclose the
full per-seed attack-success distribution rather than a single value.
"""
from __future__ import annotations
import json
import numpy as np


def aggregate(system: str, out_path: str = None):
    with open(f"results/benchmarks/{system}_final_pranet_test.json") as f:
        seeds = json.load(f)
    case_names = list(seeds[0]["adversarial"].keys())
    out = {"system": system, "cases": {}}
    for case in case_names:
        n_success_per_seed = [s["adversarial"][case].get("n_success", 0) for s in seeds]
        n_total = seeds[0]["adversarial"][case].get("n_total", 220)
        aurocs, d_errors, loc_f1s = [], [], []
        for s in seeds:
            adv = s["adversarial"][case]
            if adv.get("n_success", 0) > 0 and "detection" in adv:
                aurocs.append(adv["detection"]["auroc"])
                d_errors.append(adv["detection"]["d_error"])
                loc_f1s.append(adv["localization"]["macro_f1"])
        out["cases"][case] = dict(
            n_total=n_total,
            n_success_per_seed=n_success_per_seed,
            n_success_pooled=int(sum(n_success_per_seed)),
            attack_success_rate_mean=float(np.mean(n_success_per_seed) / n_total),
            attack_success_rate_std=float(np.std(n_success_per_seed) / n_total),
            attack_success_rate_range=[min(n_success_per_seed) / n_total, max(n_success_per_seed) / n_total],
            n_seeds_with_success=int(sum(1 for n in n_success_per_seed if n > 0)),
            detection_auroc_mean=float(np.mean(aurocs)) if aurocs else None,
            detection_auroc_std=float(np.std(aurocs)) if aurocs else None,
            detection_d_error_mean=float(np.mean(d_errors)) if d_errors else None,
            localization_macro_f1_mean=float(np.mean(loc_f1s)) if loc_f1s else None,
            localization_macro_f1_std=float(np.std(loc_f1s)) if loc_f1s else None,
        )
    out_path = out_path or f"results/benchmarks/{system}_adversarial_aggregate.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[{system}]")
    for case, d in out["cases"].items():
        print(f"  {case}: attack success {d['n_success_per_seed']}/{d['n_total']} per seed "
              f"(mean rate {d['attack_success_rate_mean']:.1%} +/- {d['attack_success_rate_std']:.1%}), "
              f"pooled n={d['n_success_pooled']}, "
              f"detection AUROC={d['detection_auroc_mean']}, loc macro-F1={d['localization_macro_f1_mean']}")
    return out


if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or ["case30", "case118"]):
        aggregate(s)
