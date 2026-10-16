"""KTAS case study, part 2: cross-site transfer and perturbation operators on real records.

Two analyses that the random-split case study (ktas_case_study.py) cannot provide.

(A) Cross-site transfer. The data come from two emergency departments (Group 1 = 688
    visits, Group 2 = 579). Each family is trained and calibrated on one site (2/3 train,
    1/3 calibration, seeded split stratified by tier and outcome) and scored on every visit
    of the other site, in both directions, over twenty seeds. The site indicator is dropped
    (it is constant within a site). Conformal coverage on the external site is reported
    unweighted and with an estimated covariate-shift weight (cross-fitted discriminator,
    Tibshirani et al. 2019), exactly as in the synthetic shift experiment.

(B) Perturbation operators on real inputs. On the same seeded 50/25/25 random splits as the
    main case study, each fitted model is scored on perturbed copies of its test fold:
    masking of the continuous triage measurements (fraction p, re-imputed with the training
    median) and Gaussian noise on them (sigma in units of the training SD). The conformal
    threshold is always the one fitted on the clean calibration fold.

All constants (tiers, thresholds, weights, alpha, bins, gates) are those of the main study.
Run:  PYTHONPATH=src python scripts/ktas_external.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ktas_case_study import (ALPHA, FAMILIES, HARM_WEIGHTS, N_BINS, N_SEEDS,  # noqa: E402
                             TIER_THRESHOLD, aurc, conformal, load)
from basics_cdss.metrics.calibration import expected_calibration_error  # noqa: E402
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402
from basics_cdss.metrics.risk_control import fnr_at_threshold  # noqa: E402
from basics_cdss.metrics.weighted_conformal import (estimate_density_ratio,  # noqa: E402
                                                    weighted_quantile)

OUT = REPO / "results"
CONT = ["Age", "Patients number per hour", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT",
        "Saturation"]


def score(p, y, tiers):
    tl = np.array([t.lower() for t in tiers])
    hi = tiers == "High"
    fw = np.array([int(pi >= TIER_THRESHOLD[t]) for pi, t in zip(p, tl)])
    return {"auroc": roc_auc_score(y, p),
            "ece_overall": expected_calibration_error(y, p, N_BINS),
            "ece_High": expected_calibration_error(y[hi], p[hi], N_BINS),
            "aurc_High": aurc(y[hi], p[hi]),
            "fnr_high_tier": fnr_at_threshold(y, p, TIER_THRESHOLD["high"], hi),
            "harm_framework": weighted_harm_loss(y, fw, tl, HARM_WEIGHTS)}


def weighted_cov(p_cal, y_cal, p_te, y_te, w_cal, w_te):
    s = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    cov, size = [], []
    for i in range(len(y_te)):
        q = weighted_quantile(s, w_cal, 1 - ALPHA, float(w_te[i]))
        in1, in0 = (1 - p_te[i]) <= q, p_te[i] <= q
        cov.append(in1 if y_te[i] == 1 else in0)
        size.append(int(in1) + int(in0))
    return float(np.mean(cov)), float(np.mean(size))


def cross_site(X, y, tiers, site):
    Xs = X.drop(columns=["regional_ed"])
    rows = []
    for src in (0, 1):
        dev, ext = np.where(site == src)[0], np.where(site != src)[0]
        strat = np.array([f"{t}_{v}" for t, v in zip(tiers[dev], y[dev])])
        for name in FAMILIES:
            for seed in range(N_SEEDS):
                tr, cal = train_test_split(dev, test_size=1 / 3, random_state=seed,
                                           stratify=strat)
                med = Xs.iloc[tr].median()
                Xm = Xs.fillna(med).fillna(0).values.astype(float)  # column all-missing at one site -> 0
                m = FAMILIES[name](seed).fit(Xm[tr], y[tr])
                p_cal = m.predict_proba(Xm[cal])[:, 1]
                p = m.predict_proba(Xm[ext])[:, 1]
                r = {"direction": f"site{src + 1}->site{2 - src}", "model": name,
                     "seed": seed, "n_ext": len(ext), **score(p, y[ext], tiers[ext])}
                r["cov_unweighted"], r["size_unweighted"] = conformal(p_cal, y[cal], p, y[ext])
                w_c, w_t = estimate_density_ratio(Xm[cal], Xm[ext], seed=seed)
                r["cov_weighted"], r["size_weighted"] = weighted_cov(
                    p_cal, y[cal], p, y[ext], w_c, w_t)
                r["ece_high_gate"] = r["ece_High"] <= 0.10
                rows.append(r)
            print("cross-site", src, name, flush=True)
    return pd.DataFrame(rows)


def perturb(Xm, cols, kind, level, med, sd, rng):
    Xp = Xm.copy()
    j = [cols.index(c) for c in CONT if c in cols]
    if kind == "mask":
        M = rng.rand(len(Xp), len(j)) < level
        sub = Xp[:, j]
        sub[M] = np.asarray(med)[j][np.where(M)[1]]
        Xp[:, j] = sub
    elif kind == "noise":
        Xp[:, j] = Xp[:, j] + rng.normal(0, level * np.asarray(sd)[j], (len(Xp), len(j)))
    return Xp


def operators(X, y, tiers):
    cols = list(X.columns)
    conds = [("clean", 0.0), ("mask", 0.2), ("mask", 0.4), ("noise", 0.15), ("noise", 0.5)]
    rows = []
    idx = np.arange(len(y))
    strat = np.array([f"{t}_{v}" for t, v in zip(tiers, y)])
    for name in FAMILIES:
        for seed in range(N_SEEDS):
            tr, tmp = train_test_split(idx, test_size=0.5, random_state=seed, stratify=strat)
            cal, te = train_test_split(tmp, test_size=0.5, random_state=seed,
                                       stratify=strat[tmp])
            med = X.iloc[tr].median()
            Xm = X.fillna(med).values.astype(float)
            sd = Xm[tr].std(axis=0)
            m = FAMILIES[name](seed).fit(Xm[tr], y[tr])
            p_cal = m.predict_proba(Xm[cal])[:, 1]
            for kind, lev in conds:
                rng = np.random.RandomState(1000 + seed)
                Xt = perturb(Xm[te], cols, kind, lev, med.values, sd, rng)
                p = m.predict_proba(Xt)[:, 1]
                r = {"model": name, "seed": seed, "condition": f"{kind} {lev:g}",
                     "accuracy": float(((p >= 0.5) == y[te]).mean()),
                     **score(p, y[te], tiers[te])}
                r["cov_unweighted"], _ = conformal(p_cal, y[cal], p, y[te])
                rows.append(r)
        print("operators", name, flush=True)
    return pd.DataFrame(rows)


def main():
    X, y, tiers = load()
    site = X["regional_ed"].values
    cs = cross_site(X, y, tiers, site)
    cs.to_csv(OUT / "ktas_cross_site_seed.csv", index=False)
    keys = ["auroc", "ece_overall", "ece_High", "aurc_High", "fnr_high_tier",
            "harm_framework", "cov_unweighted", "size_unweighted", "cov_weighted",
            "size_weighted"]
    g = cs.groupby(["direction", "model"])
    s = g[keys].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["ece_high_gate_pass_rate"] = g["ece_high_gate"].mean()
    s["seeds_unweighted_below_090"] = g["cov_unweighted"].apply(lambda v: int((v < 0.9).sum()))
    s["seeds_weighted_below_090"] = g["cov_weighted"].apply(lambda v: int((v < 0.9).sum()))
    s.reset_index().to_csv(OUT / "ktas_cross_site_summary.csv", index=False)
    print(s.round(3).T.to_string())

    op = operators(X, y, tiers)
    op.to_csv(OUT / "ktas_operator_seed.csv", index=False)
    k2 = ["accuracy", "auroc", "ece_overall", "ece_High", "fnr_high_tier", "harm_framework",
          "cov_unweighted"]
    g2 = op.groupby(["model", "condition"], sort=False)
    s2 = g2[k2].mean()
    s2["seeds_cov_below_090"] = g2["cov_unweighted"].apply(lambda v: int((v < 0.9).sum()))
    s2.reset_index().to_csv(OUT / "ktas_operator_summary.csv", index=False)
    print(s2.round(3).to_string())


if __name__ == "__main__":
    main()
