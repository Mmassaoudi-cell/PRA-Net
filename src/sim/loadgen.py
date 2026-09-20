"""
Time-varying load-scenario ("chronic") generator over a base pandapower network.

Each chronic's system-wide load trajectory is drawn from a real historical
hourly utility load time series (PJM Interconnection for the 30-bus system,
ISO New England for the 118-bus system; both public system-load records),
per-unit normalized and applied multiplicatively to the base-case P/Q loads
together with per-bus correlated AR(1) noise (capturing bus-level deviation
from the aggregate system trend), subject to a feasibility check (AC
power-flow convergence). This grounds the simulated measurement stream in
real load dynamics (diurnal/weekly cycles, ramps, non-Gaussian variability)
rather than a purely synthetic curve.

Chronics are the unit of train/val/test splitting (scenario-level split, see
DATA_SPLIT_MANIFEST.csv) to avoid the leakage that a random-row split would
introduce between adjacent, highly-correlated timesteps.
"""
from __future__ import annotations
import numpy as np
import pandapower as pp


def real_load_curve(n_steps: int, rng: np.random.Generator, load_series: np.ndarray) -> np.ndarray:
    """Sample a random contiguous window of `n_steps` hours from a real
    historical system-load series, per-unit normalize it around the window's
    own mean, and return it as a multiplicative load-scaling curve."""
    n_avail = len(load_series)
    if n_avail <= n_steps:
        idx = np.arange(n_steps) % n_avail
        window = load_series[idx]
    else:
        start = rng.integers(0, n_avail - n_steps)
        window = load_series[start:start + n_steps]
    window = window.astype(np.float64)
    pu = window / window.mean()
    drift = rng.normal(0, 0.01, size=1)
    return np.clip(pu + drift, 0.30, 1.85)


def daily_curve(n_steps: int, rng: np.random.Generator, period: int = 96) -> np.ndarray:
    """Fallback synthetic multi-harmonic curve, retained for the
    label-noise/robustness sweeps and for systems with no matched real-load
    series; the main dataset-generation path uses `real_load_curve` above."""
    t = np.arange(n_steps)
    base = (
        1.0
        + 0.25 * np.sin(2 * np.pi * t / period - 1.2)
        + 0.08 * np.sin(4 * np.pi * t / period + 0.5)
    )
    drift = rng.normal(0, 0.01, size=1)
    return np.clip(base + drift, 0.35, 1.6)


def ar1_noise(n_steps: int, n_series: int, rng: np.random.Generator, rho: float = 0.85, sigma: float = 0.035):
    x = np.zeros((n_steps, n_series))
    x[0] = rng.normal(0, sigma, size=n_series)
    for t in range(1, n_steps):
        x[t] = rho * x[t - 1] + rng.normal(0, sigma * np.sqrt(1 - rho ** 2), size=n_series)
    return x


def generate_chronic(net, n_steps: int, seed: int, load_series: np.ndarray | None = None):
    """Return dict with per-timestep bus voltage angle/magnitude ground truth
    (from AC power flow) for one chronic of n_steps snapshots. If
    `load_series` (a real historical hourly system-load array) is given, the
    system-wide load trajectory is drawn from it (see `real_load_curve`);
    otherwise falls back to the synthetic multi-harmonic curve."""
    rng = np.random.default_rng(seed)
    n_load = len(net.load)
    curve = real_load_curve(n_steps, rng, load_series) if load_series is not None else daily_curve(n_steps, rng)
    noise = ar1_noise(n_steps, n_load, rng)
    scale = curve[:, None] * (1.0 + noise)
    scale = np.clip(scale, 0.3, 1.75)

    base_p = net.load.p_mw.values.copy()
    base_q = net.load.q_mvar.values.copy()

    va_deg = np.zeros((n_steps, len(net.bus)))
    vm_pu = np.zeros((n_steps, len(net.bus)))
    p_flow = None
    q_flow = None
    p_inj = np.zeros((n_steps, len(net.bus)))
    q_inj = np.zeros((n_steps, len(net.bus)))
    ok = np.ones(n_steps, dtype=bool)

    for t in range(n_steps):
        net.load.p_mw = base_p * scale[t]
        net.load.q_mvar = base_q * scale[t]
        try:
            pp.runpp(net, algorithm="nr", init="results" if t > 0 else "flat", max_iteration=30)
        except Exception:
            ok[t] = False
            continue
        va_deg[t] = net.res_bus.va_degree.values
        vm_pu[t] = net.res_bus.vm_pu.values
        pf = net.res_line.p_from_mw.values
        qf = net.res_line.q_from_mvar.values
        if len(net.trafo) > 0:
            pf = np.concatenate([pf, net.res_trafo.p_hv_mw.values])
            qf = np.concatenate([qf, net.res_trafo.q_hv_mvar.values])
        if p_flow is None:
            p_flow = np.zeros((n_steps, len(pf)))
            q_flow = np.zeros((n_steps, len(qf)))
        p_flow[t] = pf
        q_flow[t] = qf
        p_inj[t] = net.res_bus.p_mw.values
        q_inj[t] = net.res_bus.q_mvar.values

    net.load.p_mw = base_p
    net.load.q_mvar = base_q

    return {
        "va_deg": va_deg, "vm_pu": vm_pu,
        "p_flow": p_flow, "q_flow": q_flow,
        "p_inj": p_inj, "q_inj": q_inj,
        "ok": ok,
    }
