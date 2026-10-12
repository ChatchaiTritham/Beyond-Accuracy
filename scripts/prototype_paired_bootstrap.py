"""Paired bootstrap of the prototype's harm difference, and a binomial interval for its coverage.

run_all.py reports per-tier bootstrap intervals for the framework-guided and accuracy-only harm
losses separately, which cannot say whether their difference is distinguishable from zero.
This script rebuilds the seed-42 prototype exactly as run_all.py does (same cohort, split,
classifier and tier thresholds) and resamples the test fold once per replicate for *both*
policies, so the interval is for the paired difference.

Output: results/prototype_paired_bootstrap.json
Usage:  python scripts/prototype_paired_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import run_all  # noqa: E402
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402

N_BOOT = 1000


def main() -> None:
    np.random.seed(run_all.SEED)
    X, y, _, tiers = run_all.build_cohort()
    strat = np.array([f"{t}_{l}" for t, l in zip(tiers, y)])
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.5, random_state=run_all.SEED, stratify=strat)
    cal, te = train_test_split(tmp, test_size=0.5, random_state=run_all.SEED,
                               stratify=strat[tmp])
    clf = run_all.fit_classifier(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    tl = np.array([t.lower() for t in tiers[te]])
    thr = {"low": 0.60, "medium": 0.45, "high": 0.30}
    base = (p >= 0.5).astype(int)
    fw = np.array([int(pi >= thr[t]) for pi, t in zip(p, tl)])
    w = run_all.HARM_WEIGHTS
    yt = y[te]
    h_base = weighted_harm_loss(yt, base, tl, w)
    h_fw = weighted_harm_loss(yt, fw, tl, w)
    rng = np.random.RandomState(run_all.SEED)
    diffs = []
    for _ in range(N_BOOT):
        b = rng.randint(0, len(yt), len(yt))
        diffs.append(weighted_harm_loss(yt[b], base[b], tl[b], w)
                     - weighted_harm_loss(yt[b], fw[b], tl[b], w))
    diffs = np.asarray(diffs)
    n_test = len(yt)
    k = int(round(0.9286 * n_test))
    ci = binomtest(k, n_test).proportion_ci(confidence_level=0.95, method="exact")
    out = {"harm_baseline": h_base, "harm_framework": h_fw, "diff_point": h_base - h_fw,
           "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
           "share_boot_diff_gt0": float((diffs > 0).mean()),
           "conformal_covered": k, "n_test": n_test,
           "conformal_cov_ci95_exact": [ci.low, ci.high], "n_boot": N_BOOT}
    (REPO / "results" / "prototype_paired_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
