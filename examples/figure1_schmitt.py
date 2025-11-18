import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- physical constants ----------
A = 100.0
beta = 0.1
V_hyst = 0.5  # kept for future use / clarity
N = 50_000    # samples per run
T = 0.01      # dt implicit in loop (time base)

# ---------- derived (analytic Kramers parameters) ----------
Delta_V = 0.5 * beta**2 * A          # barrier height (analytic)
omega_p = np.sqrt(1 + beta * A)      # well curvature
omega_s = np.sqrt(abs(1 - beta * A)) # saddle curvature


def schmitt_run(sigma: float) -> float:
    """
    Return the normalized flip rate for a given thermal-noise std `sigma`.

    The input is a linearly swept voltage with added Gaussian noise.
    The output is the fraction of timesteps in which the Schmitt output flips.
    """
    # input signal + noise
    V_in = np.linspace(-1, 1, N) + np.random.normal(0, sigma, N)

    V_out = 0.0   # low state
    flips = 0

    for i in range(N - 1):
        # threshold depends on current output (recursive dependence)
        V_th = beta * V_out

        # hysteresis comparison logic
        if V_out == 0.0 and V_in[i] > V_th:
            V_out = 5.0
            flips += 1
        elif V_out == 5.0 and V_in[i] < V_th:
            V_out = 0.0
            flips += 1

    # normalized flip rate
    return flips / N


def main():
    # ---------- sweep noise and compare to Kramers rate ----------
    sigma_vals = np.logspace(-2, 0, 15)  # 0.01 -> 1.0 (noise range)

    f_sim = np.array([schmitt_run(sigma) for sigma in sigma_vals])

    # Analytic Kramers escape rate
    f_kram = (omega_p * omega_s / (2 * np.pi)) * np.exp(
        -Delta_V / sigma_vals**2
    )

    # ---------- save data ----------
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "docs" / "figures"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(
        {"sigma": sigma_vals, "f_will_sim": f_sim, "f_kram": f_kram}
    )
    csv_path = out_dir / "figure1_schmitt_kramers.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved data to {csv_path}")

    # ---------- plot ----------
    plt.figure()
    plt.loglog(sigma_vals, f_sim, "o", label="simulation")
    plt.loglog(sigma_vals, f_kram, "-", label="Kramers")
    plt.xlabel("noise std σ")
    plt.ylabel("flip rate f_will")
    plt.legend()
    plt.title("Schmitt proto-will: flip rate vs noise")

    png_path = out_dir / "figure1_flip_rate_vs_noise.png"
    plt.savefig(png_path, bbox_inches="tight")
    print(f"[INFO] Saved figure to {png_path}")


if __name__ == "__main__":
    main()
