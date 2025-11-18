from serf_will.schmitt import schmitt_run
"""
Schmitt-trigger proto-will simulation core.

This module provides:
- schmitt_run(sigma): Monte-Carlo flip-rate under thermal noise
- kramers_rate(sigma): analytic Kramers escape rate for comparison

Both are used to generate Figure 1 in the SERF paper.
"""

from __future__ import annotations

import numpy as np

# ---------- core parameters ----------
# These match the values used in the SERF proto-will simulations.

A: float = 100.0      # gain
beta: float = 0.1     # feedback coefficient
V_HYST: float = 0.5   # hysteresis marker (kept for conceptual clarity)

N_DEFAULT: int = 50_000  # steps per run

# Time-step is implicit in the discrete loop; we keep it as a named constant
T_DEFAULT: float = 0.01

# ---------- derived Kramers parameters ----------
# Barrier height of the effective double-well.
Delta_V: float = 0.5 * beta**2 * A

# Curvatures near minimum (omega_p) and saddle (omega_s).
omega_p: float = np.sqrt(1.0 + beta * A)
omega_s: float = np.sqrt(abs(1.0 - beta * A))


def schmitt_run(sigma: float, n_steps: int = N_DEFAULT, seed: int | None = None) -> float:
    """
    Run a Schmitt-trigger proto-will simulation for a given noise level.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian thermal noise added to the input.
    n_steps : int, optional
        Number of timesteps in the simulation (default: N_DEFAULT).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Normalized flip rate (flips per timestep) for the given sigma.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, n_steps)
    else:
        noise = np.random.normal(0.0, sigma, n_steps)

    # Input: linear sweep plus noise
    v_in = np.linspace(-1.0, 1.0, n_steps) + noise

    v_out = 0.0  # low state
    flips = 0

    for i in range(n_steps - 1):
        # Recursive threshold depends on current output
        v_th = beta * v_out

        if v_out == 0.0 and v_in[i] > v_th:
            v_out = 5.0
            flips += 1
        elif v_out == 5.0 and v_in[i] < v_th:
            v_out = 0.0
            flips += 1

    return flips / n_steps


def kramers_rate(sigma: np.ndarray | float) -> np.ndarray:
    """
    Analytic Kramers escape rate for the corresponding Schmitt system.

    Parameters
    ----------
    sigma : float or ndarray
        Thermal noise standard deviation(s).

    Returns
    -------
    ndarray
        Theoretical flip rate(s) according to Kramers' law.
    """
    sigma = np.asarray(sigma, dtype=float)
    prefactor = omega_p * omega_s / (2.0 * np.pi)
    return prefactor * np.exp(-Delta_V / sigma**2)
