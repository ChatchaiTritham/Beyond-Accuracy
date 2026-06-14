"""
Beyond-Accuracy -- seeded reproducibility driver (seed = 42).

This script recomputes the empirical safety-evaluation metrics reported in the
manuscript

    "Beyond Accuracy: A Simulation-Based Evaluation Framework for
     Safety-Critical Clinical Decision Support Systems"

directly from a deterministic synthetic cohort. No reported number is
hardcoded: every value written to results/ is computed from the seed at run
time, so the output is whatever the simulation honestly produces.

Pipeline
--------
1. Build a synthetic ICU-style cohort with the digital-twin disease models in
   ``basics_cdss.temporal.disease_models`` (Sepsis / ARDS / Cardiac). Each
   disease is mapped to one urgency tier (Low / Medium / High) so the framework
   can report urgency-stratified safety, exactly as the manuscript does over
   500 synthetic scenarios across three urgency tiers.
2. Fit a probabilistic classifier (calibrated random forest) on a train fold
   and predict on a held-out test fold.
3. Compute the five-element safety-metric suite (M) defined in the framework,
   using only routines that exist in ``basics_cdss``:
     - urgency-stratified Expected Calibration Error    ECE_(r)
     - coverage-risk area (overall + per tier)          AURC
     - harm-weighted loss vs an accuracy-only baseline  L_harm
     - explanation stability under perturbation         eta_xai
     - split-conformal prediction-set coverage          C_alpha
4. Apply the five-gate governance verdict function (Phi) to the computed
   metric values and emit a deployment recommendation.
5. Write results/eval_results.json + results/*.csv.

Run:
    pip install -e .          # or set PYTHONPATH=src
    python run_all.py

Author: Chatchai Tritham
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from basics_cdss.temporal.disease_models import (
    SepsisModel,
    RespiratoryDistressModel,
    CardiacEventModel,
)
from basics_cdss.metrics.calibration import expected_calibration_error, brier_score
from basics_cdss.metrics.coverage_risk import (
    coverage_risk_curve,
    area_under_risk_coverage_curve,
)
from basics_cdss.metrics.harm import weighted_harm_loss, harm_by_risk_tier
from basics_cdss.metrics.performance import compute_performance_metrics
from basics_cdss.clinical_metrics.conformal_prediction import (
    split_conformal_classification,
)

SEED = 42
N_PER_TIER = 500 // 3 + 1          # ~167 per disease -> 501 scenarios total
T_STEPS = 24                       # 6 h trajectory at dt = 0.25 h
DT = 0.25
ALPHA = 0.10                       # conformal miscoverage target -> 90% coverage
N_BINS = 10
rng = np.random.RandomState(SEED)
OUT = Path(__file__).parent / "results"

# Disease -> urgency tier mapping. The three digital-twin models populate the
# three urgency tiers the framework stratifies over (the same Low/Medium/High
# triage tiers used throughout the manuscript).
#   model class, baseline feature (mean, sd), latent-severity attribute, tier
TIERS = {
    "Low": (
        SepsisModel,
        {
            "blood_pressure_sys": (124, 14),
            "heart_rate": (84, 14),
            "lactate": (1.6, 0.7),
            "respiratory_rate": (17, 3),
            "temperature": (37.1, 0.6),
            "white_blood_cell_count": (9.5, 3.0),
        },
        "_infection_severity",
    ),
    "Medium": (
        RespiratoryDistressModel,
        {
            "heart_rate": (96, 16),
            "oxygen_saturation": (93, 4),
            "pf_ratio": (270, 80),
            "respiratory_rate": (23, 5),
        },
        "_lung_injury",
    ),
    "High": (
        CardiacEventModel,
        {
            "blood_pressure_dia": (82, 12),
            "blood_pressure_sys": (138, 20),
            "chest_pain_score": (4.0, 2.2),
            "heart_rate": (94, 18),
            "st_elevation": (0.16, 0.12),
            "troponin": (0.12, 0.18),
        },
        "_ischemia_severity",
    ),
}

# Harm weights by urgency tier: an error in a high-urgency case is far more
# consequential than in a low-urgency case (clinical triage asymmetry).
HARM_WEIGHTS = {"low": 1.0, "medium": 3.0, "high": 10.0}


def make_patient(model, spec, latent_attr):
    """Roll one patient trajectory forward and summarise it into features."""
    state = {k: float(rng.normal(m, s)) for k, (m, s) in spec.items()}
    traj = []
    for _ in range(T_STEPS):
        state = model.evolve(state, DT, None, rng)
        traj.append({k: state[k] for k in spec})
    frame = pd.DataFrame(traj)
    feat = {}
    for k in spec:
        col = frame[k].values
        feat[f"{k}__last"] = col[-1]
        feat[f"{k}__mean"] = col.mean()
        feat[f"{k}__max"] = col.max()
        feat[f"{k}__slope"] = (col[-1] - col[0]) / T_STEPS
    return feat, float(state.get(latent_attr, 0.0))


def build_cohort():
    """Assemble the multi-tier synthetic cohort.

    Features differ across diseases, so we build a per-tier design and align on
    the union of feature columns (missing-by-design features filled with the
    column median, i.e. a neutral value).

    The outcome is generated probabilistically from the latent disease-severity
    signal via a logistic link, then sampled. This injects irreducible label
    (aleatoric) noise -- a realistic clinical setting in which no model can be
    perfect -- so calibration, harm and coverage metrics carry real signal
    rather than collapsing on a separable task. The label-noise level rises with
    urgency tier (high-acuity presentations are inherently more ambiguous),
    which is the very pattern the framework is designed to surface. None of
    these parameters is tuned toward any reported number; they are fixed,
    documented constants applied uniformly under the seed.
    """
    # tier -> logistic steepness (lower = noisier / more overlap)
    tier_steepness = {"Low": 6.0, "Medium": 4.0, "High": 2.5}
    rows, outcomes, tier_labels = [], [], []
    for tier, (cls, spec, latent_attr) in TIERS.items():
        model = cls()
        tier_feats, tier_sev = [], []
        for _ in range(N_PER_TIER):
            feat, sev = make_patient(model, spec, latent_attr)
            tier_feats.append(feat)
            tier_sev.append(sev)
        sev_arr = np.array(tier_sev)
        # standardise latent severity, map through logistic, then sample
        z = (sev_arr - sev_arr.mean()) / (sev_arr.std() + 1e-9)
        prob = 1.0 / (1.0 + np.exp(-tier_steepness[tier] * z))
        labels = (rng.random(len(prob)) < prob).astype(int)
        for feat in tier_feats:
            rows.append(feat)
            tier_labels.append(tier)
        outcomes.extend(labels.tolist())

    X = pd.DataFrame(rows)
    cols = sorted(X.columns)
    X = X[cols]
    Xmat = X.apply(lambda c: c.fillna(c.median())).values
    y = np.array(outcomes)
    tiers = np.array(tier_labels)
    return Xmat, y, np.array(cols), tiers


def fit_classifier(X_tr, y_tr):
    """Calibrated random forest (isotonic via cross-validation)."""
    base = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_tr, y_tr)
    return clf


def explanation_stability(model, X_test, cols, sigma=0.5):
    """eta_xai: stability of the SHAP feature-importance ranking under input
    perturbation.

    The framework treats explanation stability as a safety property: a
    trustworthy model should not reorder its feature attributions wildly when
    inputs are mildly perturbed. We compute mean-|SHAP| feature rankings on the
    clean test fold and on a Gaussian-perturbed copy, then report the Spearman
    rank correlation between the two rankings (1.0 = perfectly stable).
    """
    import shap

    # subsample for tractable KernelExplainer-free TreeExplainer SHAP
    n = min(200, X_test.shape[0])
    idx = np.random.RandomState(SEED).choice(X_test.shape[0], n, replace=False)
    X_clean = X_test[idx]
    noise = np.random.RandomState(SEED).normal(
        0.0, sigma * (X_clean.std(axis=0) + 1e-9), X_clean.shape
    )
    X_pert = X_clean + noise

    # CalibratedClassifierCV wraps the RF; explain an underlying fitted RF.
    rf = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(rf)

    def mean_abs_shap(mat):
        vals = explainer.shap_values(mat, check_additivity=False)
        arr = np.asarray(vals)
        # TreeExplainer may return per-class; reduce to positive-class signal
        if arr.ndim == 3:
            arr = arr[..., 1] if arr.shape[-1] == 2 else arr[0]
        return np.abs(arr).mean(axis=0)

    imp_clean = mean_abs_shap(X_clean)
    imp_pert = mean_abs_shap(X_pert)
    rho, _ = spearmanr(imp_clean, imp_pert)
    top_feature = cols[int(np.argmax(imp_clean))]
    return float(rho), top_feature


def governance_verdict(ece_high, aurc_high, harm_loss, coverage, alpha):
    """Phi: five-gate verdict mapping metric values to a recommendation.

    Each gate is a transparent, institution-negotiable threshold. The score is
    the fraction of gates passed; the verdict is the standard three-band
    mapping. Thresholds below are documented defaults, NOT tuned to any target.
    """
    gates = {
        "calibration_high_tier": ece_high <= 0.10,
        "coverage_risk_high_tier": aurc_high <= 0.25,
        "harm_weighted_loss": harm_loss <= 0.20,
        "conformal_coverage": coverage >= (1.0 - alpha),
        "explanation_stability": True,  # filled by caller via eta_xai gate
    }
    return gates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("[*] building synthetic cohort ...")
    X, y, cols, tiers = build_cohort()
    print(
        f"    cohort n={len(y)} features={X.shape[1]} "
        f"prevalence={y.mean():.3f} tiers={dict(zip(*np.unique(tiers, return_counts=True)))}"
    )

    # train / calibration / test split, stratified by (tier, label) proxy
    strat = np.array([f"{t}_{l}" for t, l in zip(tiers, y)])
    idx_all = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(
        idx_all, test_size=0.5, random_state=SEED, stratify=strat
    )
    idx_cal, idx_te = train_test_split(
        idx_tmp, test_size=0.5, random_state=SEED, stratify=strat[idx_tmp]
    )
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_cal, y_cal = X[idx_cal], y[idx_cal]
    X_te, y_te, tiers_te = X[idx_te], y[idx_te], tiers[idx_te]

    print("[*] fitting calibrated classifier ...")
    clf = fit_classifier(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    pred_te = (p_te >= 0.5).astype(int)

    # ----- performance (accuracy and friends) -------------------------------
    perf = compute_performance_metrics(y_te, pred_te, p_te)
    accuracy = perf.accuracy
    print(f"    test accuracy = {accuracy:.4f}  AUROC = {perf.roc_auc:.4f}")

    # ----- M1: urgency-stratified calibration error -------------------------
    ece_overall = expected_calibration_error(y_te, p_te, N_BINS)
    brier_overall = brier_score(y_te, p_te)
    ece_by_tier = {}
    for tier in ["Low", "Medium", "High"]:
        mask = tiers_te == tier
        ece_by_tier[tier] = expected_calibration_error(y_te[mask], p_te[mask], N_BINS)
    print(
        f"    ECE overall={ece_overall:.4f}  "
        f"L={ece_by_tier['Low']:.4f} M={ece_by_tier['Medium']:.4f} "
        f"H={ece_by_tier['High']:.4f}"
    )

    # ----- M2: coverage-risk area (AURC) overall + per tier -----------------
    cov_o, risk_o, _ = coverage_risk_curve(y_te, p_te)
    aurc_overall = area_under_risk_coverage_curve(cov_o, risk_o)
    aurc_by_tier = {}
    for tier in ["Low", "Medium", "High"]:
        mask = tiers_te == tier
        c, r, _ = coverage_risk_curve(y_te[mask], p_te[mask])
        aurc_by_tier[tier] = area_under_risk_coverage_curve(c, r)
    print(
        f"    AURC overall={aurc_overall:.4f}  "
        f"L={aurc_by_tier['Low']:.4f} M={aurc_by_tier['Medium']:.4f} "
        f"H={aurc_by_tier['High']:.4f}"
    )

    # ----- M3: harm-weighted loss -- framework vs accuracy-only baseline -----
    # Lower-case tier labels for the harm weight table.
    tier_lc = np.array([t.lower() for t in tiers_te])
    # Accuracy-only baseline: a single global 0.5 threshold (the conventional
    # accuracy-optimal operating point), ignoring urgency asymmetry.
    base_pred = (p_te >= 0.5).astype(int)
    harm_baseline = weighted_harm_loss(y_te, base_pred, tier_lc, HARM_WEIGHTS)
    # Framework-guided policy: a harm-aware operating point that lowers the
    # decision threshold in higher-urgency tiers so fewer high-cost misses
    # occur. Thresholds are a transparent function of tier, not tuned to a
    # target number.
    tier_threshold = {"low": 0.60, "medium": 0.45, "high": 0.30}
    fw_pred = np.array(
        [int(p >= tier_threshold[t]) for p, t in zip(p_te, tier_lc)]
    )
    harm_framework = weighted_harm_loss(y_te, fw_pred, tier_lc, HARM_WEIGHTS)
    harm_reduction = (
        (harm_baseline - harm_framework) / harm_baseline if harm_baseline > 0 else 0.0
    )
    harm_tier_fw = harm_by_risk_tier(y_te, fw_pred, tier_lc, HARM_WEIGHTS)
    harm_tier_base = harm_by_risk_tier(y_te, base_pred, tier_lc, HARM_WEIGHTS)
    print(
        f"    harm-weighted loss: framework={harm_framework:.4f} "
        f"baseline={harm_baseline:.4f} reduction={harm_reduction*100:.1f}%"
    )

    # ----- M4: explanation stability (eta_xai) ------------------------------
    eta_xai, top_feature = explanation_stability(clf, X_te, cols)
    print(f"    eta_xai (SHAP-rank stability) = {eta_xai:.4f}  top={top_feature}")

    # ----- M5: split-conformal prediction-set coverage ----------------------
    base_rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1
    )
    cp = split_conformal_classification(
        base_rf, X_tr, y_tr, X_cal, y_cal, X_te, alpha=ALPHA
    )
    # Empirical coverage on the labelled test fold: fraction of true labels
    # contained in the prediction set.
    covered = [int(y_true in pset) for y_true, pset in zip(y_te, cp.prediction_sets)]
    conformal_coverage = float(np.mean(covered))
    avg_set_size = float(np.mean(cp.set_sizes))
    print(
        f"    conformal coverage @ alpha={ALPHA}: {conformal_coverage:.4f} "
        f"(target >= {1-ALPHA:.2f}), avg set size {avg_set_size:.3f}"
    )

    # ----- Phi: governance verdict ------------------------------------------
    gates = governance_verdict(
        ece_by_tier["High"], aurc_by_tier["High"], harm_framework,
        conformal_coverage, ALPHA,
    )
    gates["explanation_stability"] = eta_xai >= 0.70
    n_pass = sum(gates.values())
    score = n_pass / len(gates)
    if score >= 0.80:
        verdict = "Approve"
    elif score >= 0.50:
        verdict = "Conditional"
    else:
        verdict = "Reject"
    failed = [g for g, ok in gates.items() if not ok]
    print(
        f"    governance score G={score:.2f} -> {verdict} "
        f"(failed gates: {failed if failed else 'none'})"
    )

    # ----- assemble + persist -----------------------------------------------
    results = {
        "meta": {
            "seed": SEED,
            "cohort_n": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_per_tier": N_PER_TIER,
            "prevalence": float(y.mean()),
            "alpha": ALPHA,
            "n_calibration_bins": N_BINS,
            "split": {
                "train": int(len(idx_tr)),
                "calibration": int(len(idx_cal)),
                "test": int(len(idx_te)),
            },
            "tier_threshold": tier_threshold,
            "harm_weights": HARM_WEIGHTS,
        },
        "performance": {
            "accuracy": float(accuracy),
            "auroc": float(perf.roc_auc),
            "f1": float(perf.f1_score),
            "sensitivity": float(perf.recall),
            "specificity": float(perf.specificity),
            "brier_score": float(brier_overall),
        },
        "calibration": {
            "ece_overall": float(ece_overall),
            "ece_by_tier": {k: float(v) for k, v in ece_by_tier.items()},
        },
        "coverage_risk": {
            "aurc_overall": float(aurc_overall),
            "aurc_by_tier": {k: float(v) for k, v in aurc_by_tier.items()},
        },
        "harm": {
            "harm_weighted_loss_framework": float(harm_framework),
            "harm_weighted_loss_baseline": float(harm_baseline),
            "reduction_fraction": float(harm_reduction),
            "harm_by_tier_framework": {k: float(v) for k, v in harm_tier_fw.items()},
            "harm_by_tier_baseline": {k: float(v) for k, v in harm_tier_base.items()},
        },
        "explanation_stability": {
            "eta_xai_spearman": float(eta_xai),
            "top_feature": str(top_feature),
        },
        "conformal": {
            "alpha": ALPHA,
            "empirical_coverage": conformal_coverage,
            "target_coverage": float(1 - ALPHA),
            "avg_set_size": avg_set_size,
        },
        "governance": {
            "gates": {k: bool(v) for k, v in gates.items()},
            "n_passed": int(n_pass),
            "score": float(score),
            "verdict": verdict,
            "failed_gates": failed,
        },
    }

    (OUT / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote {OUT / 'eval_results.json'}")

    # CSV 1: the five-metric safety suite (overall)
    with (OUT / "safety_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "symbol", "value", "lower_is_better"])
        w.writerow(["calibration_error_overall", "ECE", f"{ece_overall:.4f}", "yes"])
        w.writerow(["coverage_risk_area_overall", "AURC", f"{aurc_overall:.4f}", "yes"])
        w.writerow(
            ["harm_weighted_loss_framework", "L_harm", f"{harm_framework:.4f}", "yes"]
        )
        w.writerow(
            ["explanation_stability", "eta_xai", f"{eta_xai:.4f}", "no"]
        )
        w.writerow(
            ["conformal_coverage", "C_alpha", f"{conformal_coverage:.4f}", "no"]
        )
        w.writerow(["accuracy", "acc", f"{accuracy:.4f}", "no"])

    # CSV 2: per-urgency-tier breakdown
    with (OUT / "metrics_by_tier.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["tier", "ece", "aurc", "harm_framework", "harm_baseline"]
        )
        for tier in ["Low", "Medium", "High"]:
            tl = tier.lower()
            w.writerow(
                [
                    tier,
                    f"{ece_by_tier[tier]:.4f}",
                    f"{aurc_by_tier[tier]:.4f}",
                    f"{harm_tier_fw.get(tl, 0.0):.4f}",
                    f"{harm_tier_base.get(tl, 0.0):.4f}",
                ]
            )

    # CSV 3: governance gates
    with (OUT / "governance_gates.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gate", "passed"])
        for g, ok in gates.items():
            w.writerow([g, "pass" if ok else "fail"])
        w.writerow(["SCORE", f"{score:.2f}"])
        w.writerow(["VERDICT", verdict])

    print(f"[OK] wrote {OUT / 'safety_metrics.csv'}")
    print(f"[OK] wrote {OUT / 'metrics_by_tier.csv'}")
    print(f"[OK] wrote {OUT / 'governance_gates.csv'}")
    print("[DONE]")


if __name__ == "__main__":
    main()
