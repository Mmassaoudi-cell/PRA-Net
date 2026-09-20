"""
End-to-end dataset generation driver: builds AC-based measurement sequences,
injects BDD-passing DC-FDIAs (generic random pool + 4 canonical localization
test cases + a disjoint adversarial-training pool), assembles sliding-window
sequences (k=6), assigns a scenario/chronic-level train/val/test split, and
writes:
  data/processed/{system}/dataset.npz
  data/processed/{system}/DATA_SPLIT_MANIFEST.csv

Scale note (documented deviation from the source paper, see
SOURCE_PAPER_AUDIT.md / REPRODUCTION_REPORT.md): the paper uses 230,400
normal + 25,000 BDD-passing-FDIA raw measurement vectors. Reproducing that
exact volume via full nonlinear AC power-flow solves for every snapshot is
computationally prohibitive for this study's compute budget; we generate a
reduced-but-substantial, fully documented volume instead (see CONFIG below).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.network import build_network, dc_jacobian, adjacency_graph
from sim.loadgen import generate_chronic
from sim.fdia import make_fdia_vector, verify_bdd_passing, wls_state_estimate, bdd_threshold_empirical

K = 6  # sliding window length, matches paper

REAL_LOAD_SOURCES = {
    # Real hourly utility system-load time series driving each test system's
    # chronic generation (see sim/loadgen.py::real_load_curve). Two distinct
    # public utility datasets are used to diversify the realistic load shapes
    # driving the two test systems.
    "case30": (r"C:\Users\MMASSAOUDI\Desktop\Data\Load Data\PJM_Load_hourly.csv", "PJM_Load_MW"),
    "case118": (r"C:\Users\MMASSAOUDI\Desktop\Data\Load Data\isone_load.csv", "load"),
}


def load_real_series(system: str) -> np.ndarray:
    path, col = REAL_LOAD_SOURCES[system]
    sep = ";" if path.endswith("isone_load.csv") else ","
    df = pd.read_csv(path, sep=sep)
    series = pd.to_numeric(df[col], errors="coerce").dropna().values
    return series.astype(np.float64)


CONFIG = {
    "case30": dict(
        n_chronics=40, n_steps=700, attack_rate=0.10,
        mag_range=(0.02, 0.09),
        test_cases={
            "case1": [5, 6, 7],      # buses 6-8 (1-idx)
            "case2": [28, 29],       # buses 29-30
        },
        adv_train_buses=[0, 1, 2, 7, 8, 9, 13, 14, 15, 26, 27],
        n_case_eval=220,
    ),
    "case118": dict(
        n_chronics=25, n_steps=520, attack_rate=0.10,
        mag_range=(0.02, 0.09),
        test_cases={
            "case3": [61, 62, 63, 64, 65, 66],  # buses 62-67
            "case4": [113, 114],                 # buses 114-115
        },
        adv_train_buses=[0, 1, 2, 51, 52, 53, 62, 63, 64, 102, 103, 104],
        n_case_eval=220,
    ),
}


def build_measurement_matrix(chronic):
    return np.concatenate([
        chronic["p_flow"], chronic["q_flow"],
        chronic["p_inj"], chronic["q_inj"],
        chronic["vm_pu"],
    ], axis=1)  # (n_steps, d)


def add_noise(z_clean, rng):
    sigma = np.abs(z_clean) * 0.1 / 3.0
    sigma = np.maximum(sigma, 1e-3)
    return z_clean + rng.normal(0, sigma), sigma


def dc_slice(z, n_branch, n_bus):
    p_flow = z[:n_branch]
    p_inj = z[2 * n_branch: 2 * n_branch + n_bus]
    return np.concatenate([p_flow, p_inj])


def dc_sigma(sigma, n_branch, n_bus):
    return dc_slice(sigma, n_branch, n_bus)


def calibrate_bdd_tau(net, jac, cfg, seed: int, load_series: np.ndarray, n_calib_steps: int = 300, alpha: float = 0.01) -> float:
    """Empirically calibrate the DC-WLS BDD chi-square threshold from clean
    (un-attacked) AC-simulated residuals (see fdia.bdd_threshold_empirical)."""
    rng = np.random.default_rng(seed + 31337)
    chronic = generate_chronic(net, n_calib_steps, seed=seed + 31337, load_series=load_series)
    z_clean = build_measurement_matrix(chronic)
    z_noisy, sigma = add_noise(z_clean, rng)
    n_branch, n_bus = jac["n_branch"], jac["n_bus"]
    Js = []
    for t in range(n_calib_steps):
        if not chronic["ok"][t]:
            continue
        z_dc_noisy = dc_slice(z_noisy[t], n_branch, n_bus)
        r_inv = 1.0 / (dc_sigma(sigma[t], n_branch, n_bus) ** 2)
        _, _, J = wls_state_estimate(z_dc_noisy, jac["H_dc"], r_inv)
        Js.append(J)
    return bdd_threshold_empirical(np.array(Js), alpha=alpha)


def inject_generic_attacks(chronic_id, z_noisy, z_clean, sigma, jac, bus_pool, rng, attack_rate, tau):
    n_steps, d = z_noisy.shape
    n_branch, n_bus = jac["n_branch"], jac["n_bus"]
    attacked_labels = [[] for _ in range(n_steps)]
    z_out = z_noisy.copy()
    n_attacks = int(n_steps * attack_rate)
    attack_steps = rng.choice(n_steps, size=n_attacks, replace=False)
    for t in attack_steps:
        k = rng.integers(1, 4)
        buses = list(rng.choice(bus_pool, size=min(k, len(bus_pool)), replace=False))
        a, _ = make_fdia_vector(jac["H_dc"], jac["non_slack"], buses, rng)
        z_dc_noisy = dc_slice(z_noisy[t], n_branch, n_bus)
        r_inv = 1.0 / (dc_sigma(sigma[t], n_branch, n_bus) ** 2)
        ok = verify_bdd_passing(z_dc_noisy, a, jac["H_dc"], r_inv, tau)
        if not ok:
            continue
        z_out[t, :n_branch] += a[:n_branch]
        z_out[t, 2 * n_branch:2 * n_branch + n_bus] += a[n_branch:]
        attacked_labels[t] = [int(b) for b in buses]
    return z_out, attacked_labels


def make_case_eval_sequences(net, jac, chronic, buses, n_needed, rng, tau, k=K):
    n_steps = chronic["vm_pu"].shape[0]
    z_clean = build_measurement_matrix(chronic)
    z_noisy, sigma = add_noise(z_clean, rng)
    n_branch, n_bus = jac["n_branch"], jac["n_bus"]
    seqs = []
    tries = 0
    attempts = list(rng.permutation(np.arange(k - 1, n_steps)))
    for t in attempts:
        if len(seqs) >= n_needed:
            break
        tries += 1
        a, _ = make_fdia_vector(jac["H_dc"], jac["non_slack"], buses, rng)
        z_dc_noisy = dc_slice(z_noisy[t], n_branch, n_bus)
        r_inv = 1.0 / (dc_sigma(sigma[t], n_branch, n_bus) ** 2)
        if not verify_bdd_passing(z_dc_noisy, a, jac["H_dc"], r_inv, tau):
            continue
        window = z_noisy[t - k + 1: t + 1].copy()
        window[-1, :n_branch] += a[:n_branch]
        window[-1, 2 * n_branch:2 * n_branch + n_bus] += a[n_branch:]
        seqs.append(window)
    return np.stack(seqs) if seqs else np.zeros((0, k, z_clean.shape[1]))


def process_system(system: str, out_dir: str, seed: int = 0):
    cfg = CONFIG[system]
    rng_master = np.random.default_rng(seed)
    net = build_network(system)
    jac = dc_jacobian(net)
    graph = adjacency_graph(net)
    n_bus = jac["n_bus"]
    all_buses = list(range(n_bus))
    reserved = set(sum(cfg["test_cases"].values(), [])) | set(cfg["adv_train_buses"])
    generic_pool = [b for b in all_buses if b not in reserved]

    load_series = load_real_series(system)
    print(f"  loaded real hourly load series: {REAL_LOAD_SOURCES[system][0]} ({len(load_series)} hours)")

    tau_dc = calibrate_bdd_tau(net, jac, cfg, seed, load_series)
    print(f"  calibrated empirical DC-BDD chi-square threshold tau={tau_dc:.2f}")

    chronic_records, manifest_rows = [], []
    seq_id = 0
    X_all, y_all, buses_all, split_all, chronic_all, dcstats = [], [], [], [], [], []

    n_chronics = cfg["n_chronics"]
    splits = (["train"] * int(n_chronics * 0.70) +
              ["val"] * int(n_chronics * 0.15))
    splits += ["test"] * (n_chronics - len(splits))
    rng_master.shuffle(splits)

    for ci in range(n_chronics):
        seed_c = seed * 10000 + ci
        rng = np.random.default_rng(seed_c)
        chronic = generate_chronic(net, cfg["n_steps"], seed=seed_c, load_series=load_series)
        if not chronic["ok"].all():
            bad = (~chronic["ok"]).sum()
            print(f"  [warn] chronic {ci} had {bad} non-converging steps, dropping them")
        z_clean = build_measurement_matrix(chronic)
        z_noisy, sigma = add_noise(z_clean, rng)
        z_attacked, labels = inject_generic_attacks(
            ci, z_noisy, z_clean, sigma, jac, generic_pool, rng, cfg["attack_rate"], tau_dc)

        split = splits[ci]
        ok_mask = chronic["ok"]
        n_steps = z_attacked.shape[0]
        for t in range(K - 1, n_steps):
            if not ok_mask[t - K + 1:t + 1].all():
                continue
            window = z_attacked[t - K + 1: t + 1]
            window_labels = labels[t - K + 1: t + 1]
            atk_buses = sorted(set(sum(window_labels, [])))
            y = 1 if len(atk_buses) > 0 else 0
            X_all.append(window.astype(np.float32))
            y_all.append(y)
            buses_all.append(atk_buses)
            split_all.append(split)
            chronic_all.append(ci)
            manifest_rows.append(dict(seq_id=seq_id, system=system, chronic_id=ci,
                                       timestep=t, split=split, label=y,
                                       attacked_buses=";".join(map(str, atk_buses)),
                                       source="generic"))
            seq_id += 1
        print(f"  chronic {ci}/{n_chronics} split={split} attacks_injected={sum(1 for l in labels if l)} steps={n_steps}")

    # Canonical localization test cases: injected only into TEST-split chronics
    test_chronic_ids = [ci for ci in range(n_chronics) if splits[ci] == "test"]
    case_eval_store = {}
    for case_name, buses in cfg["test_cases"].items():
        collected = []
        per_chronic_need = max(1, cfg["n_case_eval"] // max(1, len(test_chronic_ids)) + 1)
        for ci in test_chronic_ids:
            seed_c = seed * 10000 + ci
            rng = np.random.default_rng(seed_c + 999)
            chronic = generate_chronic(net, cfg["n_steps"], seed=seed_c, load_series=load_series)
            seqs = make_case_eval_sequences(net, jac, chronic, buses, per_chronic_need, rng, tau_dc)
            collected.append(seqs)
            if sum(s.shape[0] for s in collected) >= cfg["n_case_eval"]:
                break
        arr = np.concatenate(collected, axis=0)[:cfg["n_case_eval"]] if collected else np.zeros((0, K, X_all[0].shape[-1]))
        case_eval_store[case_name] = dict(X=arr.astype(np.float32), buses=buses)
        print(f"  case-eval {case_name}: {arr.shape[0]} sequences, buses={buses}")

    # Adversarial-training pool: dedicated bus set, drawn from TRAIN chronics only
    train_chronic_ids = [ci for ci in range(n_chronics) if splits[ci] == "train"]
    adv_pool = []
    per_chronic_need = max(1, 400 // max(1, len(train_chronic_ids)) + 1)
    for ci in train_chronic_ids:
        seed_c = seed * 10000 + ci
        rng = np.random.default_rng(seed_c + 555)
        chronic = generate_chronic(net, cfg["n_steps"], seed=seed_c, load_series=load_series)
        seqs = make_case_eval_sequences(net, jac, chronic, cfg["adv_train_buses"], per_chronic_need, rng, tau_dc)
        adv_pool.append(seqs)
        if sum(s.shape[0] for s in adv_pool) >= 400:
            break
    adv_pool_arr = np.concatenate(adv_pool, axis=0)[:400] if adv_pool else np.zeros((0, K, X_all[0].shape[-1]))
    print(f"  adv-train pool: {adv_pool_arr.shape[0]} sequences, buses={cfg['adv_train_buses']}")

    X = np.stack(X_all)
    y = np.array(y_all, dtype=np.int64)
    split_arr = np.array(split_all)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "dataset.npz"),
        X=X, y=y, split=split_arr, chronic_id=np.array(chronic_all),
        d=X.shape[-1], n_branch=jac["n_branch"], n_bus=jac["n_bus"],
        adv_pool_X=adv_pool_arr,
        **{f"case_eval_{k}_X": v["X"] for k, v in case_eval_store.items()},
        **{f"case_eval_{k}_buses": np.array(v["buses"]) for k, v in case_eval_store.items()},
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(out_dir, "DATA_SPLIT_MANIFEST.csv"), index=False)

    meta = dict(system=system, d=int(X.shape[-1]), n_branch=int(jac["n_branch"]), n_bus=int(jac["n_bus"]),
                n_seq=int(X.shape[0]), pos_rate=float(y.mean()),
                split_counts={s: int((split_arr == s).sum()) for s in ["train", "val", "test"]},
                test_cases={k: v for k, v in cfg["test_cases"].items()},
                adv_train_buses=cfg["adv_train_buses"], graph_edges=list(graph.edges()))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{system}] DONE: {X.shape}, pos_rate={y.mean():.3f}, splits={meta['split_counts']}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["case30", "case118", "all"], default="all")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    systems = ["case30", "case118"] if args.system == "all" else [args.system]
    for s in systems:
        print(f"=== Generating {s} ===")
        process_system(s, os.path.join(args.out, s), seed=args.seed)
