"""The safety suite across model families and seeds, on the large cohort.

The prototype reported one calibrated random forest on 501 scenarios and called the framework
model-agnostic. Neither half of that is evidence: a single family cannot support a claim about
families, and a single seeded run cannot separate a model's behaviour from its draw. This
script runs the suite over six families and many seeds on the 6,000-scenario cohort, and adds
the decision-analytic comparison the certified alert threshold invites -- net benefit against
treat-all and treat-none, which is the question a clinician asks of any alarm.

Outputs
  results/model_seed_metrics.csv   one row per (seed, model): the five suite metrics + accuracy
  results/model_seed_summary.csv   mean / sd / range per model and metric
  results/certified_net_benefit.csv  net benefit of the LTT-certified threshold vs treat-all

Usage: python scripts/model_sweep.py [--seeds 20] [--start 42]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

import run_all  # noqa: E402
from basics_cdss.metrics.calibration import expected_calibration_error  # noqa: E402
from basics_cdss.metrics.coverage_risk import coverage_risk_curve  # noqa: E402
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402
from basics_cdss.metrics.risk_control import fnr_at_threshold, learn_then_test  # noqa: E402

N_PER_TIER = 2000
N_BINS = 10
ALPHA_RISK, DELTA = 0.05, 0.10
RESULTS = REPO / "results"

FAMILIES = {
    "logistic_regression": lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=s)),
    "random_forest": lambda s: CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1),
        method="isotonic", cv=3),
    "gradient_boosting": lambda s: GradientBoostingClassifier(random_state=s),
    "mlp": lambda s: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                        random_state=s)),
    "svm_rbf": lambda s: CalibratedClassifierCV(
        make_pipeline(StandardScaler(), SVC(random_state=s)),
        method="sigmoid", cv=3),
    "uncalibrated_forest": lambda s: RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=s, n_jobs=-1),
    # the same MLP with Platt scaling: one scalar fitted on held-out folds, which is what
    # temperature scaling does. Without this arm the MLP's calibration error cannot be told
    # apart from the absence of any calibration step.
    "mlp_calibrated": lambda s: CalibratedClassifierCV(
        make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                                      max_iter=800, random_state=s)),
        method="sigmoid", cv=3),
}


def aurc(y_true: np.ndarray, p: np.ndarray) -> float:
    """Normalised area under the risk-coverage curve.

    ``coverage_risk_curve`` returns the grid in descending coverage and can emit a
    non-finite risk at the degenerate end, so the grid is sorted and filtered before
    integration; without that the integral silently returns NaN for some models.
    """
    cov, risk, _ = coverage_risk_curve(y_true, p)
    cov = np.asarray(cov, dtype=float)
    risk = np.asarray(risk, dtype=float)
    keep = np.isfinite(cov) & np.isfinite(risk)
    cov, risk = cov[keep], risk[keep]
    if len(cov) < 2:
        return float("nan")
    order = np.argsort(cov)
    cov, risk = cov[order], risk[order]
    span = cov[-1] - cov[0]
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(risk, cov) / span)


def net_benefit(y_true: np.ndarray, flag: np.ndarray, threshold: float) -> float:
    """Vickers' net benefit of an arbitrary alert rule at a decision threshold."""
    n = len(y_true)
    tp = float(np.sum((flag == 1) & (y_true == 1)))
    fp = float(np.sum((flag == 1) & (y_true == 0)))
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def run_seed(X, y, tiers, seed: int) -> list[dict]:
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    high_te = tiers[te] == "High"
    rows = []
    for name, factory in FAMILIES.items():
        model = factory(seed)
        model.fit(X[tr], y[tr])
        p_te = model.predict_proba(X[te])[:, 1]
        pred = (p_te >= 0.5).astype(int)
        tier_lc = np.array([t.lower() for t in tiers[te]])
        rows.append({
            "seed": seed, "model": name,
            "accuracy": float((pred == y[te]).mean()),
            "ece_overall": float(expected_calibration_error(y[te], p_te, N_BINS)),
            "ece_high": float(expected_calibration_error(y[te][high_te], p_te[high_te], N_BINS)),
            "aurc_overall": aurc(y[te], p_te),
            "aurc_high": aurc(y[te][high_te], p_te[high_te]),
            "harm_loss": float(weighted_harm_loss(y[te], pred, tier_lc, run_all.HARM_WEIGHTS)),
            "fnr_high_at_0.5": fnr_at_threshold(y[te], p_te, 0.5, high_te),
        })
        rows[-1].update(tier_harm(y[te], pred, tier_lc))
    return rows


def tier_harm(y_true, pred, tier_lc) -> dict:
    """Error rate and weighted harm contribution within each risk tier.

    ``harm_{tier}`` is the tier's share of the cohort-level weighted harm loss, so the three
    add up to ``harm_loss``; ``err_{tier}`` is the plain error rate inside the tier, which is
    what shows whether the weighting is doing the work or the errors really are concentrated.
    ``harm_concentration`` is the high tier's share of total harm.
    """
    import numpy as _np
    wrong = (pred != y_true)
    n = len(y_true)
    out, total = {}, 0.0
    for tier in ("low", "medium", "high"):
        m = tier_lc == tier
        w = run_all.HARM_WEIGHTS[tier]
        contrib = float(_np.sum(wrong[m]) * w / n) if m.any() else float("nan")
        out[f"err_{tier}"] = float(wrong[m].mean()) if m.any() else float("nan")
        out[f"harm_{tier}"] = contrib
        if contrib == contrib:
            total += contrib
    out["harm_concentration"] = out["harm_high"] / total if total > 0 else float("nan")
    return out


def certified_net_benefit(X, y, tiers, seed: int) -> list[dict]:
    """Net benefit of the LTT-certified threshold against treat-all and treat-none."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    high_cal, high_te = tiers[cal] == "High", tiers[te] == "High"
    lambdas = np.round(np.arange(0.05, 0.96, 0.05), 2)

    out = []
    for name, factory in FAMILIES.items():
        model = factory(seed)
        model.fit(X[tr], y[tr])
        p_cal = model.predict_proba(X[cal])[:, 1]
        p_te = model.predict_proba(X[te])[:, 1]
        npos = int((y[cal][high_cal] == 1).sum())
        res = learn_then_test(lambdas,
                              lambda lam: fnr_at_threshold(y[cal], p_cal, lam, high_cal),
                              n_cal=npos, alpha_risk=ALPHA_RISK, delta=DELTA)
        lam = res.lambda_hat
        y_h, p_h = y[te][high_te], p_te[high_te]
        prevalence = float(y_h.mean())
        row = {"seed": seed, "model": name, "certified_lambda": lam,
               "n_cal_positives": npos, "prevalence_high": round(prevalence, 4)}
        for thr_name, thr in (("at_0.05", 0.05), ("at_0.20", 0.20), ("at_0.50", 0.50)):
            treat_all = net_benefit(y_h, np.ones_like(y_h), thr)
            row[f"nb_treat_all_{thr_name}"] = round(treat_all, 4)
            row[f"nb_treat_none_{thr_name}"] = 0.0
            if lam is not None:
                nb = net_benefit(y_h, (p_h >= lam).astype(int), thr)
                row[f"nb_certified_{thr_name}"] = round(nb, 4)
                # the paired difference is the quantity the claim rests on; keeping it per
                # seed is what lets an interval be formed instead of quoting a point estimate
                row[f"nb_delta_{thr_name}"] = round(nb - treat_all, 4)
                row[f"nb_gain_over_treat_all_{thr_name}"] = round(nb - treat_all, 4)
            else:
                row[f"nb_certified_{thr_name}"] = float("nan")
                row[f"nb_gain_over_treat_all_{thr_name}"] = float("nan")
        row["alert_rate_high"] = (float((p_h >= lam).mean()) if lam is not None
                                  else float("nan"))
        out.append(row)
    return out


def write(rows, name):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name)


def main() -> None:
    ap = argparse.ArgumentParser(description="model-family and seed sweep of the safety suite")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=42)
    args = ap.parse_args()

    run_all.N_PER_TIER = N_PER_TIER
    rows, nb_rows = [], []
    for k in range(args.seeds):
        seed = args.start + k
        np.random.seed(seed)
        run_all.SEED = seed
        X, y, cols, tiers = run_all.build_cohort()
        rows.extend(run_seed(X, y, tiers, seed))
        nb_rows.extend(certified_net_benefit(X, y, tiers, seed))
        print(f"[seed {seed}] ({k + 1}/{args.seeds}) done", flush=True)

    write(rows, "model_seed_metrics.csv")
    write(nb_rows, "certified_net_benefit.csv")

    # summary
    import collections
    agg = collections.defaultdict(list)
    metrics = ["accuracy", "ece_overall", "ece_high", "aurc_overall", "aurc_high",
               "harm_loss", "fnr_high_at_0.5",
               "err_low", "err_medium", "err_high",
               "harm_low", "harm_medium", "harm_high", "harm_concentration"]
    for r in rows:
        for m in metrics:
            agg[(r["model"], m)].append(r[m])
    summary = []
    for (model, metric), vals in agg.items():
        v = np.array(vals, dtype=float)
        summary.append({"model": model, "metric": metric, "n_seeds": len(v),
                        "mean": round(float(v.mean()), 4),
                        "sd": round(float(v.std(ddof=1)), 4),
                        "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)})
    write(summary, "model_seed_summary.csv")

    # per-tier harm, one row per family and tier, so the table can be read directly
    tier_rows = []
    for model in FAMILIES:
        for tier in ("low", "medium", "high"):
            e = np.array([r[f"err_{tier}"] for r in rows if r["model"] == model], dtype=float)
            h = np.array([r[f"harm_{tier}"] for r in rows if r["model"] == model], dtype=float)
            tier_rows.append({"model": model, "tier": tier, "n_seeds": len(e),
                              "harm_weight": run_all.HARM_WEIGHTS[tier],
                              "error_rate_mean": round(float(e.mean()), 4),
                              "error_rate_sd": round(float(e.std(ddof=1)), 4),
                              "harm_contrib_mean": round(float(h.mean()), 4),
                              "harm_contrib_sd": round(float(h.std(ddof=1)), 4)})
    write(tier_rows, "harm_by_tier.csv")

    # net benefit over treat-all: mean and a percentile interval across seeds, per family
    nb_rows_out = []
    for model in FAMILIES:
        for thr_name in ("at_0.05", "at_0.20", "at_0.50"):
            key = f"nb_delta_{thr_name}"
            vals = np.array([r[key] for r in nb_rows
                             if r["model"] == model and key in r], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            nb_rows_out.append({
                "model": model, "threshold": thr_name, "n_seeds": len(vals),
                "delta_mean": round(float(vals.mean()), 4),
                "delta_sd": round(float(vals.std(ddof=1)), 4),
                "delta_p2.5": round(float(np.percentile(vals, 2.5)), 4),
                "delta_p97.5": round(float(np.percentile(vals, 97.5)), 4),
                "seeds_with_gain": int((vals > 0).sum())})
    if nb_rows_out:
        write(nb_rows_out, "net_benefit_interval.csv")

    print("\nnet benefit over treat-all at threshold 0.20 (mean [2.5, 97.5] over seeds):")
    for r in [r for r in nb_rows_out if r["threshold"] == "at_0.20"]:
        print(f"  {r['model']:22} {r['delta_mean']:+.4f} "
              f"[{r['delta_p2.5']:+.4f}, {r['delta_p97.5']:+.4f}]  "
              f"gain in {r['seeds_with_gain']}/{r['n_seeds']} seeds")

    print("\nharm concentration in the High tier (share of weighted harm):")
    for s_ in sorted([s_ for s_ in summary if s_["metric"] == "harm_concentration"],
                     key=lambda d: -d["mean"]):
        print(f"  {s_['model']:22} {s_['mean']:.3f} +/- {s_['sd']:.3f}")

    print("\nECE (overall) by family:")
    for s in sorted([s for s in summary if s["metric"] == "ece_overall"],
                    key=lambda d: d["mean"]):
        print(f"  {s['model']:22} {s['mean']:.3f} +/- {s['sd']:.3f} "
              f"[{s['min']:.3f}, {s['max']:.3f}]")


if __name__ == "__main__":
    main()
