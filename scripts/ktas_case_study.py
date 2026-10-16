"""Real-data case study: the Beyond-Accuracy verdict pipeline on KTAS triage records.

Data
----
Moon S-H, Shim JL, Park K-S, Park C-S (2019). Triage accuracy and causes of mistriage
using the Korean Triage and Acuity Scale. PLOS ONE 14(9): e0216972.
https://doi.org/10.1371/journal.pone.0216972 -- S1 Appendix (EMR review coding data,
n = 1,267 adult ED visits, two Korean EDs). Published under CC BY 4.0, which permits
reuse and redistribution with attribution. The raw file is stored unchanged in
data/ktas_raw/ together with the S2 coding guideline.

Task
----
Outcome y = 1 if the visit ended in ward admission, ICU admission, transfer, death or
operation from the ED (Disposition 2,3,5,6,7); y = 0 for discharge or discharge against
advice (1,4). Predictors are the triage-time variables only (site, sex, age, crowding,
arrival mode, injury, mental status, pain, NRS, SBP, DBP, HR, RR, BT, SpO2). The expert
KTAS level is NOT a predictor: it defines the urgency tier used for stratification
(High = KTAS 1-2, Medium = 3, Low = 4-5). Nurse KTAS, diagnosis, length of stay and the
mistriage labels are excluded (post-triage or label-derived).

Pipeline (same as run_all.py, constants unchanged)
------------------------------------------------
tier-stratified ECE, AURC, harm-weighted loss with tier thresholds {0.60,0.45,0.30}
against a 0.5 baseline, explanation stability, split-conformal coverage (alpha=0.10),
Learn-then-Test certification of high-tier FNR <= 0.05 (delta = 0.10), five-gate verdict.
A competing "standard audit" (overall AUROC >= 0.80 and overall ECE <= 0.10) is run on
the same fits for comparison.

Run:  PYTHONPATH=src python scripts/ktas_case_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from basics_cdss.metrics.calibration import expected_calibration_error  # noqa: E402
from basics_cdss.metrics.coverage_risk import (  # noqa: E402
    area_under_risk_coverage_curve, coverage_risk_curve)
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402
from basics_cdss.metrics.risk_control import fnr_at_threshold, learn_then_test  # noqa: E402

RAW = REPO / "data" / "ktas_raw" / "S1_Appendix.xlsx"
OUT = REPO / "results"
PRIMARY_SEED = 42
N_SEEDS = 20
ALPHA, ALPHA_RISK, DELTA, N_BINS = 0.10, 0.05, 0.10, 10
HARM_WEIGHTS = {"low": 1.0, "medium": 3.0, "high": 10.0}
TIER_THRESHOLD = {"low": 0.60, "medium": 0.45, "high": 0.30}
LAMBDAS = np.round(np.arange(0.05, 0.96, 0.05), 2)
TIERS = ["Low", "Medium", "High"]

FAMILIES = {
    "logistic_regression": lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=s)),
    "random_forest": lambda s: CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1),
        method="isotonic", cv=3),
    "gradient_boosting": lambda s: GradientBoostingClassifier(random_state=s),
    "mlp_calibrated": lambda s: CalibratedClassifierCV(
        make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                                      max_iter=800, random_state=s)),
        method="sigmoid", cv=3),
}


def load():
    d = pd.read_excel(RAW)
    num = ["Age", "Patients number per hour", "NRS_pain", "SBP", "DBP", "HR", "RR",
           "BT", "Saturation"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.loc[d["Pain"] == 0, "NRS_pain"] = d.loc[d["Pain"] == 0, "NRS_pain"].fillna(0)
    X = pd.DataFrame({
        "regional_ed": (d["Group"] == 2).astype(int),
        "male": (d["Sex"] == 2).astype(int),
        "injury": (d["Injury"] == 2).astype(int),
        "pain": (d["Pain"] == 1).astype(int),
        "mental": d["Mental"].astype(float),
    })
    for k, name in {1: "walk", 2: "ems119", 3: "private_car", 4: "private_amb"}.items():
        X[f"arrival_{name}"] = (d["Arrival mode"] == k).astype(int)
    for c in num:
        X[f"{c}_missing"] = d[c].isna().astype(int)
        X[c] = d[c]
    X = X.loc[:, X.std() > 0]
    y = d["Disposition"].isin([2, 3, 5, 6, 7]).astype(int).values
    tiers = np.where(d["KTAS_expert"] <= 2, "High",
                     np.where(d["KTAS_expert"] == 3, "Medium", "Low"))
    return X, y, tiers


def impute(X, idx_fit):
    med = X.iloc[idx_fit].median()
    return X.fillna(med).values.astype(float)


def aurc(y, p):
    c, r, _ = coverage_risk_curve(y, p)
    return float(area_under_risk_coverage_curve(c, r))


def conformal(p_cal, y_cal, p_te, y_te):
    """Split conformal (LAC score 1 - p_true), finite-sample quantile."""
    pc = np.where(y_cal == 1, p_cal, 1 - p_cal)
    s = 1 - pc
    n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
    in1 = (1 - p_te) <= q
    in0 = p_te <= q
    covered = np.where(y_te == 1, in1, in0)
    return float(covered.mean()), float((in1.astype(int) + in0.astype(int)).mean())


def stability(model, X_te, X_ref_med, seed, sigma=0.5):
    """Model-agnostic attribution stability: Spearman between occlusion importances
    (mean |p - p(feature := train median)|) on clean and Gaussian-perturbed inputs."""
    rng = np.random.RandomState(seed)
    Xp = X_te + rng.normal(0, sigma * (X_te.std(axis=0) + 1e-9), X_te.shape)

    def imp(M):
        base = model.predict_proba(M)[:, 1]
        out = []
        for j in range(M.shape[1]):
            Mj = M.copy()
            Mj[:, j] = X_ref_med[j]
            out.append(np.abs(base - model.predict_proba(Mj)[:, 1]).mean())
        return np.array(out)

    return float(spearmanr(imp(X_te), imp(Xp))[0])


def evaluate(name, seed, X, y, tiers, with_shap=False, cols=None):
    idx = np.arange(len(y))
    strat = np.array([f"{t}_{v}" for t, v in zip(tiers, y)])
    tr, tmp = train_test_split(idx, test_size=0.5, random_state=seed, stratify=strat)
    cal, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=strat[tmp])
    Xm = impute(X, tr)
    model = FAMILIES[name](seed).fit(Xm[tr], y[tr])
    p_cal = model.predict_proba(Xm[cal])[:, 1]
    p = model.predict_proba(Xm[te])[:, 1]
    yt, tt = y[te], tiers[te]
    tl = np.array([t.lower() for t in tt])
    r = {"model": name, "seed": seed, "n_test": len(te), "prevalence_test": yt.mean(),
         "auroc": roc_auc_score(yt, p), "brier": float(np.mean((p - yt) ** 2)),
         "accuracy": float(((p >= 0.5) == yt).mean()),
         "ece_overall": expected_calibration_error(yt, p, N_BINS),
         "aurc_overall": aurc(yt, p)}
    for t in TIERS:
        m = tt == t
        r[f"ece_{t}"] = expected_calibration_error(yt[m], p[m], N_BINS)
        r[f"aurc_{t}"] = aurc(yt[m], p[m])
        r[f"auroc_{t}"] = roc_auc_score(yt[m], p[m]) if len(set(yt[m])) > 1 else np.nan
        r[f"n_{t}"] = int(m.sum())
        r[f"prev_{t}"] = float(yt[m].mean())
    base = (p >= 0.5).astype(int)
    fw = np.array([int(pi >= TIER_THRESHOLD[t]) for pi, t in zip(p, tl)])
    r["harm_baseline"] = weighted_harm_loss(yt, base, tl, HARM_WEIGHTS)
    r["harm_framework"] = weighted_harm_loss(yt, fw, tl, HARM_WEIGHTS)
    r["fnr_high_baseline"] = fnr_at_threshold(yt, p, 0.5, tt == "High")
    r["fnr_high_framework"] = fnr_at_threshold(yt, p, 0.30, tt == "High")
    r["conformal_coverage"], r["conformal_set_size"] = conformal(p_cal, y[cal], p, yt)
    r["eta_occlusion"] = stability(model, Xm[te], np.median(Xm[tr], axis=0), seed)
    if with_shap:
        import shap
        rf = model.calibrated_classifiers_[0].estimator
        ex = shap.TreeExplainer(rf)
        noise = np.random.RandomState(seed).normal(
            0, 0.5 * (Xm[te].std(axis=0) + 1e-9), Xm[te].shape)

        def ms(M):
            a = np.asarray(ex.shap_values(M, check_additivity=False))
            if a.ndim == 3:
                a = a[..., 1]
            return np.abs(a).mean(axis=0)
        i0 = ms(Xm[te])
        r["eta_shap"] = float(spearmanr(i0, ms(Xm[te] + noise))[0])
        r["top_feature_shap"] = cols[int(np.argmax(i0))]
    # Learn-then-Test on the high-urgency calibration stratum
    mc = tiers[cal] == "High"
    npos = int((y[cal][mc] == 1).sum())
    res = learn_then_test(LAMBDAS, lambda lam: fnr_at_threshold(y[cal], p_cal, lam, mc),
                          n_cal=npos, alpha_risk=ALPHA_RISK, delta=DELTA)
    r["ltt_n_cal_pos_high"] = npos
    r["ltt_lambda_hat"] = np.nan if res.lambda_hat is None else res.lambda_hat
    r["ltt_certified"] = res.lambda_hat is not None
    if res.lambda_hat is not None:
        r["ltt_fnr_test_high"] = fnr_at_threshold(yt, p, res.lambda_hat, tt == "High")
        r["ltt_alert_rate_high"] = float((p[tt == "High"] >= res.lambda_hat).mean())
    eta = r.get("eta_shap", r["eta_occlusion"])
    gates = {"calibration_high_tier": r["ece_High"] <= 0.10,
             "coverage_risk_high_tier": r["aurc_High"] <= 0.25,
             "harm_weighted_loss": r["harm_framework"] <= 0.20,
             "conformal_coverage": r["conformal_coverage"] >= 1 - ALPHA,
             "explanation_stability": eta >= 0.70}
    for g, ok in gates.items():
        r[f"gate_{g}"] = bool(ok)
    r["G"] = sum(gates.values()) / len(gates)
    r["verdict"] = ("Approve" if r["G"] >= 0.80 else
                    "Conditional" if r["G"] >= 0.50 else "Reject")
    r["standard_audit_pass"] = bool(r["auroc"] >= 0.80 and r["ece_overall"] <= 0.10)
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}


def main():
    X, y, tiers = load()
    cols = list(X.columns)
    print(f"n={len(y)} features={len(cols)} prevalence={y.mean():.3f}")
    for t in TIERS:
        m = tiers == t
        print(f"  {t}: n={m.sum()} prevalence={y[m].mean():.3f}")
    prim = evaluate("random_forest", PRIMARY_SEED, X, y, tiers, with_shap=True, cols=cols)
    print(json.dumps(prim, indent=1, default=str))
    rows = []
    for name in FAMILIES:
        for s in range(N_SEEDS):
            rows.append(evaluate(name, s, X, y, tiers))
        print("done", name)
    sweep = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    sweep.to_csv(OUT / "ktas_seed_metrics.csv", index=False)
    pd.DataFrame([prim]).to_csv(OUT / "ktas_primary.csv", index=False)
    cohort = [{"tier": t, "n": int((tiers == t).sum()),
               "prevalence": float(y[tiers == t].mean())} for t in TIERS]
    cohort.append({"tier": "All", "n": len(y), "prevalence": float(y.mean())})
    pd.DataFrame(cohort).to_csv(OUT / "ktas_cohort.csv", index=False)
    summarize(sweep)


def summarize(sweep):
    keys = ["auroc", "brier", "ece_overall", "ece_Low", "ece_Medium", "ece_High",
            "aurc_overall", "aurc_Low", "aurc_Medium", "aurc_High", "harm_baseline",
            "harm_framework", "fnr_high_baseline", "fnr_high_framework",
            "conformal_coverage", "conformal_set_size", "eta_occlusion", "G"]
    g = sweep.groupby("model")
    summ = g[keys].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["ltt_certified_rate"] = g["ltt_certified"].mean()
    if "ltt_fnr_test_high" in sweep:
        summ["ltt_fnr_test_high_mean"] = g["ltt_fnr_test_high"].mean()
    summ["standard_audit_pass_rate"] = g["standard_audit_pass"].mean()
    for v in ["Approve", "Conditional", "Reject"]:
        summ[f"verdict_{v}_rate"] = g["verdict"].apply(lambda s: (s == v).mean())
    for gate in [c for c in sweep.columns if c.startswith("gate_")]:
        summ[f"{gate}_pass_rate"] = g[gate].mean()
    summ.reset_index().to_csv(OUT / "ktas_summary.csv", index=False)
    print(summ.T.to_string())


if __name__ == "__main__":
    if "--summarize-only" in sys.argv:
        summarize(pd.read_csv(OUT / "ktas_seed_metrics.csv"))
    else:
        main()
