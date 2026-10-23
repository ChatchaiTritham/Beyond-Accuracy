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
import statsmodels.api as sm  # calibration slope / intercept

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

class DeepEnsemble:
    """Uncertainty baseline: five independently initialised MLPs (128-64-32, ReLU, Adam,
    early stopping) whose predicted probabilities are averaged. Deep ensembles are the
    standard non-Bayesian uncertainty baseline for tabular networks."""

    def __init__(self, seed, n_members=5):
        self.members = [make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=1000, early_stopping=True,
            random_state=seed * 100 + k)) for k in range(n_members)]

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


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
    "deep_ensemble": lambda s: DeepEnsemble(s),
}
OUTCOME = "admission"   # admission | severe | undertriage (set from the command line)


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
    if OUTCOME == "admission":        # ward, ICU, transfer, death, operation
        y = d["Disposition"].isin([2, 3, 5, 6, 7]).astype(int).values
    elif OUTCOME == "severe":         # ICU, death, operation only (n = 39)
        y = d["Disposition"].isin([3, 6, 7]).astype(int).values
    elif OUTCOME == "undertriage":    # nurse level less urgent than expert level (n = 131)
        y = (d["KTAS_RN"] > d["KTAS_expert"]).astype(int).values
    else:
        raise ValueError(OUTCOME)
    tiers = np.where(d["KTAS_expert"] <= 2, "High",
                     np.where(d["KTAS_expert"] == 3, "Medium", "Low"))
    nurse = np.where(d["KTAS_RN"] <= 2, "High",
                     np.where(d["KTAS_RN"] == 3, "Medium", "Low"))
    if OUTCOME == "undertriage":
        # the expert level defines the label, so only the nurse tier may stratify
        tiers = nurse
    return X, y, tiers, nurse


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


def ece_null(p, n_bins, reps=500, seed=0):
    """Expected binned ECE of a *perfectly calibrated* model with these predicted
    probabilities (consistency resampling: y ~ Bernoulli(p)). Reported beside the
    observed ECE because the plug-in estimator is biased upward in small strata."""
    rng = np.random.RandomState(seed)
    vals = [expected_calibration_error((rng.random(len(p)) < p).astype(int), p, n_bins)
            for _ in range(reps)]
    return float(np.mean(vals)), float(np.quantile(vals, 0.95))


def calib_slope_intercept(y, p):
    """Logistic calibration slope and intercept (Cox): logit(P(y)) = a + b * logit(p)."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(pc / (1 - pc))
    if len(set(y)) < 2:
        return float("nan"), float("nan")
    try:
        slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params
        itc = sm.Logit(y, np.ones_like(lp)[:, None], offset=lp).fit(disp=0).params
        return float(slope[1]), float(itc[0])
    except Exception:
        return float("nan"), float("nan")


def evaluate(name, seed, X, y, tiers, with_shap=False, cols=None, nurse=None):
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
        r[f"ece_null_{t}"], r[f"ece_null95_{t}"] = ece_null(p[m], N_BINS, seed=seed)
        r[f"ece_excess_{t}"] = r[f"ece_{t}"] - r[f"ece_null_{t}"]
        r[f"calib_slope_{t}"], r[f"calib_intercept_{t}"] = calib_slope_intercept(yt[m], p[m])
    r["ece_null_overall"], r["ece_null95_overall"] = ece_null(p, N_BINS, seed=seed)
    r["ece_excess_overall"] = r["ece_overall"] - r["ece_null_overall"]
    r["calib_slope_overall"], r["calib_intercept_overall"] = calib_slope_intercept(yt, p)
    base = (p >= 0.5).astype(int)
    fw = np.array([int(pi >= TIER_THRESHOLD[t]) for pi, t in zip(p, tl)])
    r["harm_baseline"] = weighted_harm_loss(yt, base, tl, HARM_WEIGHTS)
    r["harm_framework"] = weighted_harm_loss(yt, fw, tl, HARM_WEIGHTS)
    r["fnr_high_baseline"] = fnr_at_threshold(yt, p, 0.5, tt == "High")
    r["fnr_high_framework"] = fnr_at_threshold(yt, p, 0.30, tt == "High")
    hi = tt == "High"
    r["alert_rate_high_baseline"] = float((p[hi] >= 0.5).mean())
    r["alert_rate_high_framework"] = float((p[hi] >= 0.30).mean())
    neg = hi & (yt == 0)
    r["spec_high_baseline"] = float((p[neg] < 0.5).mean()) if neg.sum() else np.nan
    r["spec_high_framework"] = float((p[neg] < 0.30).mean()) if neg.sum() else np.nan
    r["harm_increased"] = bool(r["harm_framework"] > r["harm_baseline"])
    # asymmetric harm: a missed positive costs the tier weight, a false alert costs 1
    w = np.array([HARM_WEIGHTS[t] for t in tl])
    def asym(pred):
        fn = (pred == 0) & (yt == 1)
        fp = (pred == 1) & (yt == 0)
        return float((w * fn + 1.0 * fp).mean())
    r["harm_asym_baseline"] = asym(base)
    r["harm_asym_framework"] = asym(fw)
    # deployable policy: thresholds chosen with the NURSE tier, scored on the expert tier
    if nurse is not None:
        nl = np.array([t.lower() for t in nurse[te]])
        fw_n = np.array([int(pi >= TIER_THRESHOLD[t]) for pi, t in zip(p, nl)])
        r["fnr_high_nurse"] = float(((fw_n == 0) & (yt == 1))[hi].sum() / max(((yt == 1) & hi).sum(), 1))
        r["alert_rate_high_nurse"] = float(fw_n[hi].mean())
        r["spec_high_nurse"] = float((fw_n[neg] == 0).mean()) if neg.sum() else np.nan
        r["harm_nurse"] = weighted_harm_loss(yt, fw_n, tl, HARM_WEIGHTS)
        r["harm_asym_nurse"] = asym(fw_n)
        r["nurse_expert_agreement"] = float((nurse[te] == tt).mean())
    # alert burden in the middle tier, where the alert changes the most decisions
    md = tt == "Medium"
    r["alert_rate_medium_framework"] = float((p[md] >= 0.45).mean())
    r["ppv_medium_framework"] = float(yt[md & (p >= 0.45)].mean()) if (md & (p >= 0.45)).sum() else np.nan
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
    X, y, tiers, nurse = load()
    cols = list(X.columns)
    suf = "" if OUTCOME == "admission" else f"_{OUTCOME}"
    print(f"outcome={OUTCOME} n={len(y)} features={len(cols)} prevalence={y.mean():.3f}")
    for t in TIERS:
        m = tiers == t
        print(f"  {t}: n={m.sum()} prevalence={y[m].mean():.3f}")
    OUT.mkdir(exist_ok=True)
    if OUTCOME == "admission":
        prim = evaluate("random_forest", PRIMARY_SEED, X, y, tiers, with_shap=True, cols=cols, nurse=nurse)
        print(json.dumps(prim, indent=1, default=str))
        pd.DataFrame([prim]).to_csv(OUT / "ktas_primary.csv", index=False)
    rows = []
    for name in FAMILIES:
        for s in range(N_SEEDS):
            rows.append(evaluate(name, s, X, y, tiers, nurse=nurse))
        print("done", name, flush=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / f"ktas_seed_metrics{suf}.csv", index=False)
    cohort = [{"tier": t, "n": int((tiers == t).sum()),
               "prevalence": float(y[tiers == t].mean())} for t in TIERS]
    cohort.append({"tier": "All", "n": len(y), "prevalence": float(y.mean())})
    pd.DataFrame(cohort).to_csv(OUT / f"ktas_cohort{suf}.csv", index=False)
    summarize(sweep, suf)


def summarize(sweep, suf=""):
    keys = ["auroc", "brier", "ece_overall", "ece_Low", "ece_Medium", "ece_High",
            "aurc_overall", "aurc_Low", "aurc_Medium", "aurc_High", "harm_baseline",
            "harm_framework", "fnr_high_baseline", "fnr_high_framework",
            "conformal_coverage", "conformal_set_size", "eta_occlusion", "G",
            "ece_null_High", "ece_excess_High", "calib_slope_High", "calib_intercept_High",
            "ece_null_overall", "ece_excess_overall", "calib_slope_overall",
            "alert_rate_high_baseline", "alert_rate_high_framework",
            "spec_high_baseline", "spec_high_framework",
            "alert_rate_medium_framework", "ppv_medium_framework", "ltt_n_cal_pos_high",
            "harm_asym_baseline", "harm_asym_framework", "fnr_high_nurse", "alert_rate_high_nurse",
            "spec_high_nurse", "harm_nurse", "harm_asym_nurse", "nurse_expert_agreement",
            "auroc_High", "prev_High", "n_High"]
    keys = [k for k in keys if k in sweep.columns]
    g = sweep.groupby("model")
    summ = g[keys].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["ltt_certified_rate"] = g["ltt_certified"].mean()
    if "ltt_fnr_test_high" in sweep:
        summ["ltt_fnr_test_high_mean"] = g["ltt_fnr_test_high"].mean()
    summ["standard_audit_pass_rate"] = g["standard_audit_pass"].mean()
    summ["harm_increased_rate"] = g["harm_increased"].mean()
    summ["ece_high_gate_pass_rate_excess"] = g["ece_excess_High"].apply(lambda s: (s <= 0.05).mean())
    for v in ["Approve", "Conditional", "Reject"]:
        summ[f"verdict_{v}_rate"] = g["verdict"].apply(lambda s: (s == v).mean())
    for gate in [c for c in sweep.columns if c.startswith("gate_")]:
        summ[f"{gate}_pass_rate"] = g[gate].mean()
    summ.reset_index().to_csv(OUT / f"ktas_summary{suf}.csv", index=False)
    print(summ.T.to_string())


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("--outcome="):
            OUTCOME = a.split("=", 1)[1]
    if "--summarize-only" in sys.argv:
        suf = "" if OUTCOME == "admission" else f"_{OUTCOME}"
        summarize(pd.read_csv(OUT / f"ktas_seed_metrics{suf}.csv"), suf)
    else:
        main()
