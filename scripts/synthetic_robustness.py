"""Synthetic sweep, part 2: boosting libraries, Mondrian conformal and tier-wise Learn-then-Test.

Same 6,000-scenario cohorts, seeds and 50/25/25 splits as model_sweep.py.

(A) XGBoost and LightGBM are added to the seven families of model_sweep.py; every family is
    scored on the suite metrics of Table 5.
(B) Marginal split conformal against Mondrian (tier-conditional) split conformal, which takes
    the quantile within each risk tier; coverage and set size are reported per tier.
(C) Tier-wise Learn-then-Test: an alert threshold is certified separately in each tier for
    FNR <= 0.05. Run at delta/3 per tier (Bonferroni), the three certificates hold jointly at
    level delta. The held-out FNR and alert rate are reported per tier.

Usage: python scripts/synthetic_robustness.py [--seeds 20] [--start 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import model_sweep as ms  # noqa: E402
import run_all  # noqa: E402
from basics_cdss.metrics.calibration import expected_calibration_error  # noqa: E402
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402
from basics_cdss.metrics.risk_control import fnr_at_threshold, learn_then_test  # noqa: E402

ALPHA = 0.10
TIERS = ("Low", "Medium", "High")
LAMBDAS = np.round(np.arange(0.05, 0.96, 0.05), 2)
FAMILIES = dict(ms.FAMILIES)
FAMILIES["xgboost"] = lambda s: XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=s, n_jobs=4, eval_metric="logloss")
FAMILIES["lightgbm"] = lambda s: LGBMClassifier(
    n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=s, n_jobs=4, verbose=-1)


def lac_q(s, alpha=ALPHA):
    n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def sets(p_cal, y_cal, g_cal, p_te, g_te):
    s = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    if g_cal is None:
        q = np.full(len(p_te), lac_q(s))
    else:
        qs = {g: lac_q(s[g_cal == g]) for g in TIERS}
        q = np.array([qs[g] for g in g_te])
    return (1 - p_te) <= q, p_te <= q


def run_seed(X, y, tiers, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    tc, tt, yt = tiers[cal], tiers[te], y[te]
    tl = np.array([t.lower() for t in tt])
    rows = []
    for name, factory in FAMILIES.items():
        m = factory(seed).fit(X[tr], y[tr])
        p_cal, p = m.predict_proba(X[cal])[:, 1], m.predict_proba(X[te])[:, 1]
        hi = tt == "High"
        pred = (p >= 0.5).astype(int)
        r = {"seed": seed, "model": name, "accuracy": float((pred == yt).mean()),
             "ece_overall": float(expected_calibration_error(yt, p, ms.N_BINS)),
             "ece_high": float(expected_calibration_error(yt[hi], p[hi], ms.N_BINS)),
             "aurc_overall": ms.aurc(yt, p),
             "harm_loss": float(weighted_harm_loss(yt, pred, tl, run_all.HARM_WEIGHTS)),
             "fnr_high_at_0.5": fnr_at_threshold(yt, p, 0.5, hi)}
        for lab, gc, gt in (("marginal", None, None), ("mondrian", tc, tt)):
            in1, in0 = sets(p_cal, y[cal], gc, p, gt)
            cov = np.where(yt == 1, in1, in0)
            size = in1.astype(int) + in0.astype(int)
            for t in TIERS:
                k = tt == t
                r[f"cov_{lab}_{t}"] = float(cov[k].mean())
                r[f"size_{lab}_{t}"] = float(size[k].mean())
        for t in TIERS:
            mc, mt = tc == t, tt == t
            npos = int((y[cal][mc] == 1).sum())
            res = learn_then_test(LAMBDAS, lambda lam: fnr_at_threshold(y[cal], p_cal, lam, mc),
                                  n_cal=npos, alpha_risk=ms.ALPHA_RISK, delta=ms.DELTA / 3)
            lam = res.lambda_hat
            r[f"ltt_npos_{t}"] = npos
            r[f"ltt_lambda_{t}"] = np.nan if lam is None else lam
            r[f"ltt_certified_{t}"] = lam is not None
            r[f"ltt_fnr_test_{t}"] = (np.nan if lam is None
                                      else fnr_at_threshold(yt, p, lam, mt))
            r[f"ltt_alert_{t}"] = np.nan if lam is None else float((p[mt] >= lam).mean())
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=42)
    a = ap.parse_args()
    run_all.N_PER_TIER = ms.N_PER_TIER
    rows = []
    for k in range(a.seeds):
        seed = a.start + k
        np.random.seed(seed)
        run_all.SEED = seed
        X, y, _, tiers = run_all.build_cohort()
        rows.extend(run_seed(X, y, np.asarray(tiers), seed))
        print(f"[seed {seed}] done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ms.RESULTS / "synthetic_robustness_seed.csv", index=False)
    num = d.drop(columns=["seed"]).groupby("model")
    s = num.mean(numeric_only=True).add_suffix("_mean").join(
        num.std(numeric_only=True).add_suffix("_sd"))
    for t in TIERS:
        s[f"ltt_certified_rate_{t}"] = num[f"ltt_certified_{t}"].mean()
    s.reset_index().round(4).to_csv(ms.RESULTS / "synthetic_robustness_summary.csv", index=False)
    print(num.mean(numeric_only=True).round(3).T.to_string())


if __name__ == "__main__":
    main()
