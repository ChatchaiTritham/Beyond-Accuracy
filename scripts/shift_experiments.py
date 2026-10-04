"""Finite-sample guarantees under the shift the framework tests with.

Two experiments, both of which the simulated cohort makes falsifiable:

1. Coverage under covariate shift. Split conformal is run on shifted test sets three ways --
   unweighted, weighted with the oracle density ratio, and weighted with a ratio estimated by
   classification. Under tilted sampling the oracle ratio is exact, so this is a direct test
   of whether the unweighted guarantee the framework previously claimed actually holds in the
   regime it is applied to. Output: results/shift_coverage.csv.

2. Risk control. Learn-then-Test certifies an alert threshold whose false-negative rate on the
   high-urgency stratum is below alpha_risk with probability 1-delta, and the certificate is
   then checked the only way it can be: repeat select-then-deploy over many simulated
   deployments and count violations, which must not exceed delta.
   Outputs: results/risk_control.csv, results/risk_control_validation.csv.

Usage:  python scripts/shift_experiments.py [--deployments 1000] [--seed 42]
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

from sklearn.model_selection import train_test_split  # noqa: E402

import run_all  # noqa: E402
from basics_cdss.metrics.weighted_conformal import (  # noqa: E402
    estimate_density_ratio,
    weighted_split_conformal,
)
from basics_cdss.metrics.risk_control import (  # noqa: E402
    fnr_at_threshold,
    learn_then_test,
)

N_PER_TIER = 2000     # 6,000 scenarios: the main driver's 501 leave the bounds vacuous
ALPHA = 0.10          # conformal miscoverage target
ALPHA_RISK = 0.05     # the FNR to certify on the high-urgency stratum
DELTA = 0.10          # error probability of the certificate
TILTS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
RESULTS = REPO / "results"


def tilted_sample(rng, z: np.ndarray, n: int, tilt: float) -> np.ndarray:
    """Indices drawn with probability proportional to exp(tilt * z).

    This is the shift whose density ratio is known in closed form: selecting a point with
    probability proportional to exp(tilt*z(x)) means dP_test/dP_cal(x) is proportional to
    exp(tilt*z(x)), with the normalising constant cancelling in the weighted quantile.
    """
    p = np.exp(tilt * z)
    p = p / p.sum()
    return rng.choice(len(z), size=n, replace=True, p=p)


def coverage_experiment(seed: int) -> list[dict]:
    run_all.N_PER_TIER = N_PER_TIER
    X, y, cols, tiers = run_all.build_cohort()
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.6, random_state=seed, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    model = run_all.fit_classifier(X[tr], y[tr])

    # Tilt along the first principal direction of the covariates: a one-dimensional,
    # reproducible severity proxy that does not privilege any hand-picked feature.
    Xc = X - X[tr].mean(axis=0)
    u, s, vt = np.linalg.svd(Xc[tr], full_matrices=False)
    direction = vt[0]
    z_all = Xc @ direction
    z_all = (z_all - z_all[tr].mean()) / (z_all[tr].std() + 1e-12)

    rng = np.random.default_rng(seed)
    rows = []
    for tilt in TILTS:
        sel = te[tilted_sample(rng, z_all[te], len(te), tilt)]
        X_te, y_te = X[sel], y[sel]

        w_cal_oracle = np.exp(tilt * z_all[cal])
        w_te_oracle = np.exp(tilt * z_all[sel])
        w_cal_est, w_te_est = estimate_density_ratio(X[cal], X_te, seed=seed)

        arms = [
            ("unweighted", None, None),
            ("oracle_weighted", w_cal_oracle, w_te_oracle),
            ("estimated_weighted", w_cal_est, w_te_est),
        ]
        for name, wc, wt in arms:
            res = weighted_split_conformal(model, X[cal], y[cal], X_te, y_te,
                                           alpha=ALPHA, weights_cal=wc, weights_test=wt,
                                           weighting=name)
            rows.append(res.as_row(tilt=tilt, target_coverage=1 - ALPHA,
                                   shift_mechanism="tilted_sampling", seed=seed))
            print(f"  tilt {tilt:>4.1f}  {name:<19} coverage {res.coverage:.3f}  "
                  f"|C| {res.avg_set_size:.2f}")
    return rows


def risk_control_experiment(seed: int, n_deployments: int) -> tuple[list[dict], list[dict]]:
    run_all.N_PER_TIER = N_PER_TIER
    X, y, cols, tiers = run_all.build_cohort()
    high = tiers == "High"
    lambdas = np.round(np.arange(0.05, 0.96, 0.05), 2)  # safest (low threshold) first

    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.6, random_state=seed, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    model = run_all.fit_classifier(X[tr], y[tr])
    p_cal = model.predict_proba(X[cal])[:, 1]
    p_te = model.predict_proba(X[te])[:, 1]

    mask_cal, mask_te = high[cal], high[te]
    n_eff = int((y[cal][mask_cal] == 1).sum())
    assert n_eff > 0, "calibration split has no high-urgency positive case"

    res = learn_then_test(
        lambdas, lambda lam: fnr_at_threshold(y[cal], p_cal, lam, mask_cal),
        n_cal=n_eff, alpha_risk=ALPHA_RISK, delta=DELTA)

    main_rows = [{
        "lambda": float(lam),
        "empirical_fnr_cal": float(r),
        "p_value": float(p),
        "certified": bool(lam in set(res.valid_lambdas.tolist())),
        "fnr_test_at_lambda": fnr_at_threshold(y[te], p_te, lam, mask_te),
        "alert_rate_high_tier": float((p_te[mask_te] >= lam).mean()),
        "specificity_high_tier": float(
            (p_te[mask_te & (y[te] == 0)] < lam).mean()
            if (mask_te & (y[te] == 0)).sum() else float("nan")),
        "alpha_risk": ALPHA_RISK, "delta": DELTA, "n_cal_positives": n_eff, "seed": seed,
    } for lam, r, p in zip(lambdas, res.risks, res.p_values)]
    print(f"  selected lambda = {res.lambda_hat} "
          f"({len(res.valid_lambdas)}/{len(lambdas)} certified)")

    # The certificate is a statement about repetitions, so repeat.
    rng = np.random.default_rng(seed)
    violations = selected = 0
    held_risks, alert_rates = [], []
    p_all = model.predict_proba(X)[:, 1]
    for d in range(n_deployments):
        s = int(rng.integers(0, 2**31 - 1))
        cal_d, te_d = train_test_split(rest, test_size=0.5, random_state=s,
                                       stratify=y[rest])
        pc, pt = p_all[cal_d], p_all[te_d]
        npos = int((y[cal_d][high[cal_d]] == 1).sum())
        if npos == 0:
            continue
        r_d = learn_then_test(lambdas,
                              lambda lam: fnr_at_threshold(y[cal_d], pc, lam, high[cal_d]),
                              n_cal=npos, alpha_risk=ALPHA_RISK, delta=DELTA)
        if r_d.lambda_hat is None:
            continue
        selected += 1
        held = fnr_at_threshold(y[te_d], pt, r_d.lambda_hat, high[te_d])
        held_risks.append(held)
        alert_rates.append(float((pt[high[te_d]] >= r_d.lambda_hat).mean()))
        violations += held > ALPHA_RISK
        if (d + 1) % 100 == 0:
            print(f"  deployment {d + 1}/{n_deployments}: "
                  f"violation rate {violations / max(selected, 1):.3f}", flush=True)

    val_rows = [{
        "n_deployments": n_deployments,
        "n_with_selection": selected,
        "violations": violations,
        "violation_rate": violations / max(selected, 1),
        "delta": DELTA,
        "alpha_risk": ALPHA_RISK,
        "mean_heldout_fnr": float(np.mean(held_risks)) if held_risks else float("nan"),
        "mean_alert_rate_high_tier": float(np.mean(alert_rates)) if alert_rates else float("nan"),
        "p95_heldout_fnr": float(np.quantile(held_risks, 0.95)) if held_risks else float("nan"),
        "guarantee_holds": bool(violations / max(selected, 1) <= DELTA),
        "seed": seed,
    }]
    return main_rows, val_rows


def write(rows: list[dict], name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name)


def main() -> None:
    ap = argparse.ArgumentParser(description="covariate-shift coverage and risk control")
    ap.add_argument("--deployments", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    print("[1/2] conformal coverage under tilted covariate shift")
    write(coverage_experiment(args.seed), "shift_coverage.csv")

    print(f"[2/2] Learn-then-Test risk control ({args.deployments} simulated deployments)")
    main_rows, val_rows = risk_control_experiment(args.seed, args.deployments)
    write(main_rows, "risk_control.csv")
    write(val_rows, "risk_control_validation.csv")
    v = val_rows[0]
    print(f"violation rate {v['violation_rate']:.3f} vs delta {DELTA} -> "
          f"{'holds' if v['guarantee_holds'] else 'VIOLATED'}")


if __name__ == "__main__":
    main()
