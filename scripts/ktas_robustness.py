"""KTAS case study, part 3: robustness analyses requested in review.

(A) Deployable tier. The main study chooses tier thresholds with the EXPERT re-triage
    KTAS level, which is not known at triage time. Here the same thresholds
    {0.60, 0.45, 0.30} are chosen with the NURSE triage level (KTAS_RN, recorded at
    triage), while outcomes and harm are still scored on the expert tier.
(B) Mondrian (tier-conditional) split conformal, stratified by the nurse tier (deployable)
    and by the expert tier (oracle), against marginal split conformal; coverage is reported
    overall and on expert KTAS 1-2 visits.
(C) Two further model families, XGBoost and LightGBM, run through the unchanged verdict
    pipeline of ktas_case_study.evaluate.
(D) Alert rates in the cross-site transfer (share of visits flagged at the tier
    threshold, overall and among expert KTAS 1-2 visits).

Splits, seeds and constants are those of ktas_case_study.py / ktas_external.py.
Run:  PYTHONPATH=src python scripts/ktas_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import ktas_case_study as kc  # noqa: E402
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402
from basics_cdss.metrics.risk_control import fnr_at_threshold  # noqa: E402

OUT = REPO / "results"
EXTRA = {
    "xgboost": lambda s: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=s,
                                       n_jobs=1, eval_metric="logloss"),
    "lightgbm": lambda s: LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                         random_state=s, n_jobs=1, verbose=-1),
}
kc.FAMILIES.update(EXTRA)


def nurse_tiers():
    d = pd.read_excel(kc.RAW)
    k = d["KTAS_RN"].values
    return np.where(k <= 2, "High", np.where(k == 3, "Medium", "Low"))


def lac_q(s, alpha=kc.ALPHA):
    n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def conformal_sets(p_cal, y_cal, g_cal, p_te, g_te):
    """Return (in1, in0) membership; g_* = None gives marginal split conformal."""
    s = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    if g_cal is None:
        q = np.full(len(p_te), lac_q(s))
    else:
        qs = {g: lac_q(s[g_cal == g]) for g in np.unique(g_cal)}
        q = np.array([qs.get(g, np.inf) for g in g_te])
    return (1 - p_te) <= q, p_te <= q


def random_split_rows(X, y, tex, tnu):
    rows = []
    idx = np.arange(len(y))
    strat = np.array([f"{t}_{v}" for t, v in zip(tex, y)])
    for name in kc.FAMILIES:
        for seed in range(kc.N_SEEDS):
            tr, tmp = train_test_split(idx, test_size=0.5, random_state=seed, stratify=strat)
            cal, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=strat[tmp])
            Xm = kc.impute(X, tr)
            m = kc.FAMILIES[name](seed).fit(Xm[tr], y[tr])
            p_cal, p = m.predict_proba(Xm[cal])[:, 1], m.predict_proba(Xm[te])[:, 1]
            yt, ex, nu = y[te], tex[te], tnu[te]
            hi = ex == "High"
            exl = np.array([t.lower() for t in ex])
            r = {"model": name, "seed": seed}
            for lab, tt in (("expert", ex), ("nurse", nu)):
                thr = np.array([kc.TIER_THRESHOLD[t.lower()] for t in tt])
                yhat = (p >= thr).astype(int)
                r[f"fnr_high_{lab}_policy"] = float(1 - yhat[hi & (yt == 1)].mean())
                r[f"harm_{lab}_policy"] = weighted_harm_loss(yt, yhat, exl, kc.HARM_WEIGHTS)
                r[f"alert_high_{lab}_policy"] = float(yhat[hi].mean())
            r["fnr_high_single05"] = fnr_at_threshold(yt, p, 0.5, hi)
            r["harm_single05"] = weighted_harm_loss(yt, (p >= 0.5).astype(int), exl, kc.HARM_WEIGHTS)
            for lab, gc, gt in (("marginal", None, None), ("mondrian_nurse", tnu[cal], nu),
                                ("mondrian_expert", tex[cal], ex)):
                in1, in0 = conformal_sets(p_cal, y[cal], gc, p, gt)
                cov = np.where(yt == 1, in1, in0)
                r[f"cov_{lab}"] = float(cov.mean())
                r[f"cov_high_{lab}"] = float(cov[hi].mean())
                r[f"size_high_{lab}"] = float((in1.astype(int) + in0.astype(int))[hi].mean())
            rows.append(r)
        print("random-split", name, flush=True)
    return pd.DataFrame(rows)


def cross_site_alerts(X, y, tex):
    Xs = X.drop(columns=["regional_ed"])
    site = X["regional_ed"].values
    rows = []
    for src in (0, 1):
        dev, ext = np.where(site == src)[0], np.where(site != src)[0]
        strat = np.array([f"{t}_{v}" for t, v in zip(tex[dev], y[dev])])
        for name in kc.FAMILIES:
            for seed in range(kc.N_SEEDS):
                tr, _ = train_test_split(dev, test_size=1 / 3, random_state=seed, stratify=strat)
                Xm = Xs.fillna(Xs.iloc[tr].median()).fillna(0).values.astype(float)
                p = kc.FAMILIES[name](seed).fit(Xm[tr], y[tr]).predict_proba(Xm[ext])[:, 1]
                hi = tex[ext] == "High"
                thr = np.array([kc.TIER_THRESHOLD[t.lower()] for t in tex[ext]])
                rows.append({"direction": f"site{src + 1}->site{2 - src}", "model": name,
                             "seed": seed, "alert_rate_all": float((p >= thr).mean()),
                             "alert_rate_high": float((p[hi] >= 0.30).mean())})
        print("cross-site", src, flush=True)
    return pd.DataFrame(rows)


def main():
    X, y, tex = kc.load()
    tnu = nurse_tiers()
    print("nurse==expert tier agreement:", float((tnu == tex).mean()))
    rs = random_split_rows(X, y, tex, tnu)
    rs.to_csv(OUT / "ktas_robustness_seed.csv", index=False)
    rs.groupby("model").agg(["mean", "std"]).round(4).to_csv(OUT / "ktas_robustness_summary.csv")
    ev = pd.DataFrame([kc.evaluate(n, s, X, y, tex) for n in EXTRA for s in range(kc.N_SEEDS)])
    ev.to_csv(OUT / "ktas_extra_families_seed.csv", index=False)
    keys = ["auroc", "ece_overall", "ece_High", "aurc_High", "fnr_high_baseline",
            "fnr_high_framework", "harm_framework", "conformal_coverage", "eta_occlusion", "G"]
    s = ev.groupby("model")[keys].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    g = ev.groupby("model")
    s["ltt_certified_rate"] = g["ltt_certified"].mean()
    s["standard_audit_pass_rate"] = g["standard_audit_pass"].mean()
    s["ece_high_gate_rate"] = g["gate_calibration_high_tier"].mean()
    for v in ["Approve", "Conditional", "Reject"]:
        s[f"verdict_{v}_rate"] = g["verdict"].apply(lambda x: (x == v).mean())
    s.reset_index().to_csv(OUT / "ktas_extra_families_summary.csv", index=False)
    cs = cross_site_alerts(X, y, tex)
    cs.to_csv(OUT / "ktas_cross_site_alerts_seed.csv", index=False)
    cs.groupby(["direction", "model"])[["alert_rate_all", "alert_rate_high"]].mean().round(4) \
      .reset_index().to_csv(OUT / "ktas_cross_site_alerts_summary.csv", index=False)
    print(rs.groupby("model").mean(numeric_only=True).round(3).T.to_string())
    print(s.round(3).T.to_string())


if __name__ == "__main__":
    main()
