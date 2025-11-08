# SERF — Formalizing Emergent Will from Recursive Contradiction

Reference implementation and reproducible figures for the Sustainability in Emergent Recursion Framework (SERF).
This repo contains:
- **Schmitt trigger** Monte‑Carlo and Kramers baseline that reproduce Figure 1 (flip‑rate vs noise).
- **RL stubs** showing the R_int = η|δ|C signal used in recursive agents.
- Ethical and citation scaffolding (MIT + Ethical Use Addendum).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/figure1_schmitt.py
```

The script saves `figure1_flip_rate_vs_noise.png`. You should see simulation points collapse onto the Kramers line across orders of magnitude.

## Citing

Please cite the software using `CITATION.cff`. The companion paper is included under `docs/` (or link a DOI when available).

## Project structure

```
src/serf_will/       # library code
examples/            # runnable demos (figures)
docs/                # paper and notes
LICENSE              # MIT
ETHICAL_USE.md       # non‑binding norms
CITATION.cff
.zenodo.json         # Zenodo metadata
```

## Contributing

Issues and PRs are welcome. See `CODE_OF_CONDUCT.md`. By contributing you agree to the MIT License and to act in the spirit of `ETHICAL_USE.md`.