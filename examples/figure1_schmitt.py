import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serf_will.schmitt import schmitt_run

def main():
    outdir = Path("examples/output")
    outdir.mkdir(parents=True, exist_ok=True)

    sigmas = np.logspace(-1, 0.5, 8)
    results = []

    for s in sigmas:
        fr = schmitt_run(sigma=s, n_steps=50_000)
        results.append((s, fr))

    df = pd.DataFrame(results, columns=["sigma", "flip_rate"])
    df.to_csv(outdir / "figure1_simulation.csv", index=False)

    from serf_will.schmitt import kramers_rate
    analytic = [kramers_rate(s) for s in sigmas]

    plt.figure(figsize=(6,4))
    plt.loglog(df["sigma"], df["flip_rate"], "o", label="Simulation")
    plt.loglog(sigmas, analytic, "-", label="Kramers")
    plt.xlabel("Thermal noise σ")
    plt.ylabel("Flip rate f_will")
    plt.legend()
    plt.title("Figure 1. Flip Rate vs Noise")
    plt.grid(which="both", ls=":")
    plt.tight_layout()
    plt.savefig(outdir / "figure1_schmitt.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
