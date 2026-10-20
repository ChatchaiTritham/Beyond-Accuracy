"""KTAS case study, part 4: a published-style competing protocol and paired statistics.

(A) Conformal selective prediction with cost-aware deferral, in the style of Kwon & Kim
    (2026, Sci Rep 16:10016): group-conditional (Mondrian) split conformal by NURSE tier;
    a visit whose set contains both labels is deferred to a clinician; on the remaining
    visits a single decision threshold is chosen on the calibration fold to minimise the
    tier-weighted cost (weights 1/3/10 on errors, deferral cost c_defer per visit).
    Reported on expert KTAS 1-2 test visits: missed admissions (not deferred and predicted
    0 while admitted), deferral rate, alert rate (flagged or deferred), and coverage
    conditional on the nurse tier used for calibration.
(B) Seed-paired Wilcoxon test of harm under tier thresholds versus a single 0.5 threshold
    (results/ktas_seed_metrics.csv).

Run:  PYTHONPATH=src python scripts/ktas_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import ktas_case_study as kc  # noqa: E402
import ktas_robustness as kr  # noqa: E402  (adds XGBoost/LightGBM to kc.FAMILIES)

OUT = REPO / "results"
C_DEFER = 0.5
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
W = kc.HARM_WEIGHTS


def cost(y, p, tl, thr, defer):
    yhat = (p >= thr).astype(int)
    err = (yhat != y) & ~defer
    return (np.array([W[t] for t in tl]) * err).mean() + C_DEFER * defer.mean()


def main():
    X, y, tex = kc.load()
    tnu = kr.nurse_tiers()
    idx = np.arange(len(y))
    strat = np.array([f"{t}_{v}" for t, v in zip(tex, y)])
    rows = []
    for name in kc.FAMILIES:
        for seed in range(kc.N_SEEDS):
            tr, tmp = train_test_split(idx, test_size=0.5, random_state=seed, stratify=strat)
            cal, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=strat[tmp])
            Xm = kc.impute(X, tr)
            m = kc.FAMILIES[name](seed).fit(Xm[tr], y[tr])
            p_cal, p = m.predict_proba(Xm[cal])[:, 1], m.predict_proba(Xm[te])[:, 1]
            in1c, in0c = kr.conformal_sets(p_cal, y[cal], tnu[cal], p_cal, tnu[cal])
            in1, in0 = kr.conformal_sets(p_cal, y[cal], tnu[cal], p, tnu[te])
            dcal, dte = in1c & in0c, in1 & in0
            tlc = np.array([t.lower() for t in tex[cal]])
            thr = min(GRID, key=lambda t: cost(y[cal], p_cal, tlc, t, dcal))
            yt, hi = y[te], tex[te] == "High"
            flag = (p >= thr) | dte
            cov = np.where(yt == 1, in1, in0)
            r = {"model": name, "seed": seed, "threshold": thr,
                 "fnr_high": float(1 - flag[hi & (yt == 1)].mean()),
                 "defer_rate_all": float(dte.mean()), "defer_rate_high": float(dte[hi].mean()),
                 "alert_rate_high": float(flag[hi].mean())}
            for t in kc.TIERS:
                r[f"cov_nurse_{t}"] = float(cov[tnu[te] == t].mean())
            rows.append(r)
        print("baseline", name, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "ktas_deferral_baseline_seed.csv", index=False)
    d.groupby("model").mean(numeric_only=True).round(4).to_csv(OUT / "ktas_deferral_baseline_summary.csv")
    print(d.groupby("model").mean(numeric_only=True).round(3).T.to_string())

    s = pd.read_csv(OUT / "ktas_seed_metrics.csv")
    out = []
    for name, g in s.groupby("model"):
        diff = g["harm_baseline"] - g["harm_framework"]
        out.append({"model": name, "median_diff": float(diff.median()),
                    "rel_reduction_mean": float((diff / g["harm_baseline"]).mean()),
                    "wilcoxon_p": float(wilcoxon(g["harm_baseline"], g["harm_framework"]).pvalue),
                    "seeds_lower": int((diff > 0).sum())})
    pd.DataFrame(out).to_csv(OUT / "ktas_harm_paired_test.csv", index=False)
    print(pd.DataFrame(out).round(4).to_string())


if __name__ == "__main__":
    main()
