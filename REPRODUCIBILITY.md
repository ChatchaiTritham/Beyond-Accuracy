# Reproducibility

This note records exactly how the numbers in this repository are produced, what
reproduces deterministically, and where the repository run intentionally differs
from the manuscript prototype. The committed `results/` artifacts are the source
of truth; no headline value is hardcoded in a figure or report.

## How to run

```bash
git clone https://github.com/ChatchaiTritham/Beyond-Accuracy.git
cd Beyond-Accuracy
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py          # fixed seed 42; a few minutes on a standard workstation
```

`run_all.py` seeds NumPy globally and passes `random_state=42` into every split
and estimator (synthetic cohort, calibrated random forest, SHAP perturbation
draw, conformal split). Rerunning under the same library versions reproduces the
metrics in `results/eval_results.json` and the CSVs exactly. The driver accepts
`--seed`; the committed artifacts are the `--seed 42` run.

## Tests

```bash
python -m pytest -q                 # fast suite: asserts the committed artifacts
python -m pytest -m slow -q         # opt-in: re-runs run_all.py and compares to committed JSON
```

The default suite reads the committed `results/` artifacts and locks the verified
seed-42 properties (governance verdict, monotonic per-tier degradation, key
metric values to tolerance). The `slow` test re-runs the pipeline end to end and
checks the regenerated metrics against the committed JSON; it skips gracefully if
the scientific stack (`numpy`/`sklearn`/`shap`/`basics_cdss`) is unavailable.

## What reproduces (seed 42)

Read directly from `results/eval_results.json` and `results/*.csv`:

- Cohort: 501 scenarios (167 per urgency tier), 48 features, split 250 / 125 / 126.
- Accuracy 0.810; AUROC 0.925.
- ECE overall 0.080; by tier 0.083 (Low) / 0.135 (Medium) / 0.163 (High) — monotonic.
- AURC overall 0.146; by tier 0.101 / 0.116 / 0.223 — monotonic.
- Harm-weighted loss 0.992 (framework) vs 1.159 (accuracy-only baseline): a 14.4% reduction.
- Explanation-ranking stability (Spearman) 0.975.
- Split-conformal coverage 0.929 against a 0.90 target (α = 0.10); mean set size 1.23.
- Governance verdict: **Conditional, G = 0.60** — 3 of 5 gates pass; high-tier
  calibration and harm-weighted loss fail.

## What does not reproduce here (and why)

The repository run shares the framework, metric definitions, verdict logic, and
the headline verdict (Conditional, G = 0.60) with the manuscript prototype, but
uses a **different cohort, model family, and harm-weight scale**. Point estimates
therefore differ in magnitude (e.g. harm weights 1/3/10 here vs 0.10/0.30/1.00 in
the prototype). The qualitative findings are the same in both: calibration and
coverage-risk degrade monotonically Low → High, and a tier-aware threshold lowers
harm without retraining. The per-quantity differences are itemised in the
"Relationship to the manuscript" table in `README.md`. The repository value is the
one to rely on; the manuscript value is reported for its tuned prototype.

The SHAP-based `explanation_stability` step requires the `shap` package; every
other metric depends only on NumPy, pandas, scikit-learn, and SciPy.

## Environment

See `requirements.txt` for pinned dependencies. Reproduction is exact on the same
library versions; minor last-digit drift can occur across major BLAS/scikit-learn
versions but does not change any reported figure at the precision quoted above.
