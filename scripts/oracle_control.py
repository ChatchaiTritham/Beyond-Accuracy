"""Positive control for the verdict rule base.

Runs the generator's own probabilities P(y = 1 | scenario) -- the Bayes-optimal
classifier for the simulator -- through the same metrics and gates as the model
families. A release gate should be able to Approve the oracle; where it cannot, the
threshold, not the model, is the binding constraint. Writes results/oracle_control.csv
(one row per seed) and results/oracle_control_summary.csv.

Usage: python scripts/oracle_control.py [--seeds 20] [--start 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import run_all  # noqa: E402
from basics_cdss.metrics.calibration import expected_calibration_error  # noqa: E402
from basics_cdss.metrics.coverage_risk import (  # noqa: E402
    area_under_risk_coverage_curve, coverage_risk_curve)
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402
from basics_cdss.metrics.risk_control import fnr_at_threshold, learn_then_test  # noqa: E402

N_PER_TIER = 2000
ALPHA, ALPHA_RISK, DELTA, N_BINS = 0.10, 0.05, 0.10, 10
HARM_WEIGHTS = {"low": 1.0, "medium": 3.0, "high": 10.0}
TIER_THRESHOLD = {"low": 0.60, "medium": 0.45, "high": 0.30}
LAMBDAS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def aurc(y, p):
    c, r, _ = coverage_risk_curve(y, p)
    return float(area_under_risk_coverage_curve(c, r))


def conformal(p_cal, y_cal, p_te, y_te):
    s = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
    in1, in0 = (1 - p_te) <= q, p_te <= q
    covered = np.where(y_te == 1, in1, in0)
    return float(covered.mean()), float((in1.astype(int) + in0.astype(int)).mean())


def one_seed(seed: int) -> dict:
    run_all.rng = np.random.RandomState(seed)
    run_all.N_PER_TIER = N_PER_TIER
    X, y, cols, tiers = run_all.build_cohort()
    p_true = np.array(run_all.LAST_TRUE_PROB)
    assert len(p_true) == len(y)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.6, random_state=seed, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    p, yt, tt = p_true[te], y[te], tiers[te]
    tl = np.array([t.lower() for t in tt])
    hi = tt == "High"
    r = {"seed": seed, "n_test": int(len(te)),
         "accuracy": float(((p >= 0.5) == yt).mean()),
         "ece_overall": expected_calibration_error(yt, p, N_BINS),
         "ece_high": expected_calibration_error(yt[hi], p[hi], N_BINS),
         "aurc_overall": aurc(yt, p), "aurc_high": aurc(yt[hi], p[hi])}
    fw = np.array([int(pi >= TIER_THRESHOLD[t]) for pi, t in zip(p, tl)])
    base = (p >= 0.5).astype(int)
    r["harm_tier_thresholds"] = weighted_harm_loss(yt, fw, tl, HARM_WEIGHTS)
    r["harm_single_threshold"] = weighted_harm_loss(yt, base, tl, HARM_WEIGHTS)
    # Bayes error rate per tier (the irreducible part) and the harm it implies
    bayes_err = np.minimum(p, 1 - p)
    r["bayes_error_high"] = float(bayes_err[hi].mean())
    r["bayes_error_overall"] = float(bayes_err.mean())
    w = np.array([HARM_WEIGHTS[t] for t in tl])
    r["bayes_harm"] = float((w * bayes_err).mean())
    r["mean_tier_weight"] = float(w.mean())
    r["fnr_high_single"] = fnr_at_threshold(yt, p, 0.5, hi)
    r["fnr_high_tier"] = fnr_at_threshold(yt, p, 0.30, hi)
    r["conformal_coverage"], r["conformal_set_size"] = conformal(p_true[cal], y[cal], p, yt)
    mc = tiers[cal] == "High"
    npos = int((y[cal][mc] == 1).sum())
    res = learn_then_test(LAMBDAS, lambda lam: fnr_at_threshold(y[cal], p_true[cal], lam, mc),
                          n_cal=npos, alpha_risk=ALPHA_RISK, delta=DELTA)
    r["ltt_certified"] = res.lambda_hat is not None
    r["ltt_lambda_hat"] = np.nan if res.lambda_hat is None else float(res.lambda_hat)
    r["ltt_alert_rate_high"] = (float((p[hi] >= res.lambda_hat).mean())
                                if res.lambda_hat is not None else np.nan)
    gates = {"calibration_high_tier": r["ece_high"] <= 0.10,
             "coverage_risk_high_tier": r["aurc_high"] <= 0.25,
             "harm_weighted_loss": r["harm_tier_thresholds"] <= 0.20,
             "conformal_coverage": r["conformal_coverage"] >= 1 - ALPHA}
    for g, ok in gates.items():
        r[f"gate_{g}"] = bool(ok)
    r["gates_passed_of4"] = int(sum(gates.values()))
    # explanation stability is not defined for the oracle (no attribution model);
    # verdict is reported with the four computable gates, XAI treated as passed
    g5 = (sum(gates.values()) + 1) / 5
    r["G_xai_assumed_pass"] = g5
    r["verdict_xai_assumed_pass"] = ("Approve" if g5 >= 0.80 else
                                     "Conditional" if g5 >= 0.50 else "Reject")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=42)
    a = ap.parse_args()
    rows = [one_seed(s) for s in range(a.start, a.start + a.seeds)]
    df = pd.DataFrame(rows)
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "oracle_control.csv", index=False)
    num = df.drop(columns=["seed"]).select_dtypes("number")
    summ = pd.concat([num.mean().rename("mean"), num.std().rename("std")], axis=1)
    summ["pass_rate"] = df[[c for c in df.columns if c.startswith("gate_")]].mean()
    summ.to_csv(out / "oracle_control_summary.csv")
    print(summ.to_string())
    print("verdicts:", df["verdict_xai_assumed_pass"].value_counts().to_dict())


if __name__ == "__main__":
    main()
