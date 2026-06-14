# Beyond-Accuracy — A Simulation-Based Evaluation Framework for Safety-Critical CDSS

Companion code for the manuscript:

> **Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical
> Clinical Decision Support Systems**
> (target: *Machine Learning*, Springer)

The paper develops a pre-deployment safety evaluation framework that scores a
clinical AI on calibration and harm-aware metrics — with formal guarantees — to
surface failure modes before patient contact, rather than relying on aggregate
accuracy.

## Scope of this repository

This repository provides a **seeded, self-contained reproducibility driver**
(`run_all.py`, seed 42) that recomputes the framework's safety-evaluation
metrics from a deterministic synthetic cohort, plus the figure-generation
scripts behind the manuscript diagrams. The simulation/metrics library
(`basics_cdss`) is vendored under `src/` so the experiment runs without any
external checkout; it is also maintained as the companion project
[BASICS-CDSS](https://github.com/ChatchaiTritham/BASICS-CDSS).

`run_all.py` builds a 501-scenario cohort across three urgency tiers
(Low / Medium / High) using the digital-twin disease models, fits a calibrated
classifier, and **computes** the five-element safety-metric suite plus the
governance verdict. No reported number is hardcoded: every value in `results/`
is computed from the seed at run time.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .          # or: pip install -r requirements.txt && set PYTHONPATH=src
python run_all.py
```

Outputs are written to `results/`:

| File | Contents |
|------|----------|
| `results/eval_results.json` | Full metric report (performance, calibration, coverage-risk, harm, explanation stability, conformal coverage, governance verdict) |
| `results/safety_metrics.csv` | The five-metric safety suite (overall) |
| `results/metrics_by_tier.csv` | Per-urgency-tier ECE / AURC / harm breakdown |
| `results/governance_gates.csv` | Five-gate verdict function outcome |

### Headline results (seed 42, recomputed by `run_all.py`)

Cohort n = 501 (167 per tier), 48 features; train/cal/test = 250/125/126.

| Metric | Symbol | Value |
|--------|--------|-------|
| Accuracy (test) | acc | 0.810 |
| AUROC | — | 0.925 |
| Calibration error, overall | ECE | 0.080 |
| Calibration error by tier | ECE(L / M / H) | 0.083 / 0.135 / 0.163 |
| Coverage-risk area, overall | AURC | 0.146 |
| Coverage-risk area by tier | AURC(L / M / H) | 0.101 / 0.116 / 0.223 |
| Harm-weighted loss (framework vs baseline) | L_harm | 0.992 vs 1.159 (−14.4%) |
| Explanation stability (SHAP-rank, Spearman) | eta_xai | 0.975 |
| Conformal coverage at alpha = 0.10 | C_alpha | 0.929 (target ≥ 0.90) |
| **Governance verdict** | Phi | **Conditional (G = 0.60)** |

The governance run reproduces the manuscript's qualitative findings: calibration
and coverage-risk **rise monotonically with urgency tier** (the framework's
central claim), conformal coverage holds above its 90% target, and the verdict
is **Conditional (G = 0.60)** with two failing gates — high-tier calibration and
harm-weighted loss — acting as named remediation targets. Absolute values differ
from the manuscript's prototype because this cohort is generated from the public
disease models with a fixed, untuned label-noise schedule rather than the
manuscript's dizziness decision-support prototype.

## Figures

```bash
python figures/generate_figures.py
```

| Path | Contents |
|------|----------|
| `run_all.py` | Seeded reproducibility driver (computes `results/`) |
| `src/basics_cdss/` | Vendored simulation + metrics library |
| `figures/generate_figures.py` | Main manuscript figures |
| `figures/generate_figures_drawio.py` | draw.io source export for diagrams |
| `figures/graphical_abstract.py` | Graphical abstract composition |

## Citation

```bibtex
@article{tritham_beyond_accuracy,
  title  = {Beyond Accuracy: A Simulation-Based Evaluation Framework for
            Safety-Critical Clinical Decision Support Systems},
  author = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Licensed under the MIT License (see `LICENSE`).
