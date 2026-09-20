"""
BDD-passing FDIA construction (classical, DC-linearized, Liu et al. 2011 /
source-paper ref [3] model) and the chi-square bad-data-detection test used
to certify that an injected attack is indeed "BDD-passing" before it is
used as a training/eval example.

Attack model: z_dc' = z_dc + H_dc @ c, where c is a state-bias vector
supported only on the attacked (non-slack) buses. Because a = H_dc @ c lies
exactly in the column space of H_dc, the WLS residual of z_dc' is identical
to that of z_dc for a purely linear system -> the chi-square BDD statistic
is unaffected regardless of the noise realization, i.e. the attack is
unobservable by construction (matches Eq. 5 of the source paper: the
attack's Jacobian-consistent sparsity pattern).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def wls_state_estimate(z_dc: np.ndarray, H_dc: np.ndarray, R_inv_diag: np.ndarray):
    W = R_inv_diag
    HtW = H_dc.T * W[None, :]
    G = HtW @ H_dc
    theta_hat = np.linalg.solve(G + 1e-8 * np.eye(G.shape[0]), HtW @ z_dc)
    r = z_dc - H_dc @ theta_hat
    J = float(r @ (W * r))
    return theta_hat, r, J


def bdd_threshold(dof: int, alpha: float = 0.01) -> float:
    return float(stats.chi2.ppf(1 - alpha, dof))


def bdd_threshold_empirical(J_samples: np.ndarray, alpha: float = 0.01) -> float:
    """
    Empirically calibrated BDD threshold from clean (un-attacked) residuals.

    The theoretical chi2(dof) threshold assumes the DC-linearized measurement
    model is exact; here it is a deliberate approximation of true AC power
    flow (documented in SOURCE_PAPER_AUDIT.md), so the WLS residual carries
    additional linearization "model-mismatch" variance beyond meter noise.
    We therefore calibrate the BDD threshold empirically as the (1-alpha)
    percentile of clean-data residuals (the same empirical-percentile
    principle the source paper itself uses for its localization threshold,
    Sec. IV-B / Table VI), rather than relying on the meter-noise-only chi2
    reference distribution.
    """
    return float(np.percentile(J_samples, 100 * (1 - alpha)))


def make_fdia_vector(H_dc: np.ndarray, non_slack: list[int], attacked_buses: list[int],
                      rng: np.random.Generator, mag_range=(0.02, 0.09)) -> np.ndarray:
    """Return the additive attack vector a = H_dc @ c (length = n_branch+n_bus)
    for a state-bias vector c supported on `attacked_buses` (radians)."""
    n_red = H_dc.shape[1]
    c = np.zeros(n_red)
    idx_map = {b: i for i, b in enumerate(non_slack)}
    for b in attacked_buses:
        if b in idx_map:
            mag = rng.uniform(*mag_range) * rng.choice([-1.0, 1.0])
            c[idx_map[b]] = mag
    a = H_dc @ c
    return a, c


def verify_bdd_passing(z_dc_clean_noisy: np.ndarray, a: np.ndarray, H_dc: np.ndarray,
                        R_inv_diag: np.ndarray, tau: float) -> bool:
    """tau must be a pre-calibrated threshold (see bdd_threshold_empirical)."""
    _, _, J_clean = wls_state_estimate(z_dc_clean_noisy, H_dc, R_inv_diag)
    _, _, J_attacked = wls_state_estimate(z_dc_clean_noisy + a, H_dc, R_inv_diag)
    return (J_clean < tau) and (abs(J_attacked - J_clean) < 1e-6 * max(1.0, abs(J_clean)))
