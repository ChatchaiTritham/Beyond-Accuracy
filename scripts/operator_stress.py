"""Clean-versus-perturbed evaluation with the four perturbation operators, and harm-weight sensitivity.

The family sweep (model_sweep.py) scores every model on clean held-out scenarios. Class C2 of the
completeness proposition is defined on the *change* in AURC when inputs are perturbed, and the
sensitivity protocol asks how the verdict responds to rising perturbation intensity. Neither can be
read off a clean sweep. This script trains each family on clean data exactly as model_sweep.py
does (same seeds, same 50/25/25 split) and then scores the same fitted model on perturbed copies
of the test fold:

  mask     each feature removed independently with probability p (package MaskOperator,
           applied in standardised feature space); a removed value is imputed with the training
           median, which is what the pipeline does with missing-by-design features
  noise    package NoiseOperator, additive N(0, sigma^2) in standard-deviation units
  conflict with probability q a scenario has one randomly chosen feature replaced by the value
           of the same feature in a random training scenario of the opposite outcome
           (a numeric analogue; the package ConflictOperator acts on categorical findings only)
  degrade  every feature coarsened to the median of its training-quantile bin (k bins);
           a numeric analogue of the package DegradeOperator, which acts on text terms only

The conformal threshold is always fitted on the clean calibration fold, so the coverage reported
on perturbed inputs is the unweighted, non-exchangeable case the paper warns about.

It also re-weights the clean-sweep harm loss under alternative tier-weight ladders, using the
per-tier error counts, to show how much of the family ranking depends on the 1/3/10 choice.

Outputs
  results/operator_stress_seed.csv     one row per (seed, model, operator, level)
  results/operator_stress_summary.csv  mean / sd over seeds, gate verdict counts
  results/harm_weight_sensitivity.csv  harm loss and family rank under five weight ladders

Usage: python scripts/operator_stress.py [--seeds 20] [--start 42]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.model_selection import train_test_split  # noqa: E402

import run_all  # noqa: E402
from model_sweep import FAMILIES, N_BINS, N_PER_TIER, aurc  # noqa: E402
from basics_cdss.metrics.calibration import expected_calibration_error  # noqa: E402
from basics_cdss.metrics.harm import weighted_harm_loss  # noqa: E402
from basics_cdss.scenario.perturbations import (  # noqa: E402
    MaskOperator, NoiseOperator, PerturbationConfig)

RESULTS = REPO / "results"
ALPHA = 0.10
OPERATORS = [("clean", 0.0),
             ("mask", 0.10), ("mask", 0.20), ("mask", 0.30), ("mask", 0.40),
             ("noise", 0.15), ("noise", 0.30), ("noise", 0.50),
             ("conflict", 0.10), ("conflict", 0.30),
             ("degrade", 4), ("degrade", 2)]
LADDERS = {"1/1/1": (1, 1, 1), "1/2/5": (1, 2, 5), "1/3/10": (1, 3, 10),
           "1/5/20": (1, 5, 20), "1/10/100": (1, 10, 100)}


def perturb(op, level, Xte, Xtr, ytr, seed):
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    med = np.median(Xtr, axis=0)
    Z = (Xte - mu) / sd
    if op == "clean":
        return Xte.copy()
    if op in ("mask", "noise"):
        keys = [f"f{j}" for j in range(Z.shape[1])]
        cfg = PerturbationConfig(p_mask=level, noise_sigma=level, protected_features=set())
        oper = (MaskOperator if op == "mask" else NoiseOperator)(cfg, seed=seed)
        out = np.empty_like(Z)
        for i, row in enumerate(Z):
            feats, _ = oper.apply(dict(zip(keys, row.tolist())))
            for j, k in enumerate(keys):
                out[i, j] = feats[k] if k in feats else (med[j] - mu[j]) / sd[j]
        return out * sd + mu
    if op == "conflict":
        X = Xte.copy()
        hit = rng.random(len(X)) < level
        for i in np.where(hit)[0]:
            j = rng.integers(X.shape[1])
            donors = np.where(ytr != ytr[rng.integers(len(ytr))])[0]
            X[i, j] = Xtr[rng.choice(donors), j]
        return X
    if op == "degrade":
        X = Xte.copy()
        for j in range(X.shape[1]):
            edges = np.unique(np.quantile(Xtr[:, j], np.linspace(0, 1, int(level) + 1)))
            b = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1, 0, len(edges) - 2)
            tb = np.clip(np.searchsorted(edges, Xtr[:, j], side="right") - 1, 0, len(edges) - 2)
            meds = np.array([np.median(Xtr[tb == k, j]) if (tb == k).any() else edges[k]
                             for k in range(len(edges) - 1)])
            X[:, j] = meds[b]
        return X
    raise ValueError(op)


def conformal_coverage(p_cal, y_cal, p_te, y_te):
    s_cal = 1.0 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    n = len(s_cal)
    q = np.quantile(s_cal, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
    s_te = 1.0 - np.where(y_te == 1, p_te, 1 - p_te)
    return float((s_te <= q).mean())


def run_seed(X, y, tiers, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
    tier_lc = np.array([t.lower() for t in tiers[te]])
    high = tiers[te] == "High"
    pert = {(op, lv): perturb(op, lv, X[te], X[tr], y[tr], seed + 7)
            for op, lv in OPERATORS}
    rows, wrows = [], []
    for name, factory in FAMILIES.items():
        model = factory(seed)
        model.fit(X[tr], y[tr])
        p_cal = model.predict_proba(X[cal])[:, 1]
        for (op, lv), Xp in pert.items():
            p = model.predict_proba(Xp)[:, 1]
            pred = (p >= 0.5).astype(int)
            r = {"seed": seed, "model": name, "operator": op, "level": lv,
                 "accuracy": float((pred == y[te]).mean()),
                 "ece_high": float(expected_calibration_error(y[te][high], p[high], N_BINS)),
                 "aurc_overall": aurc(y[te], p),
                 "aurc_high": aurc(y[te][high], p[high]),
                 "harm_loss": float(weighted_harm_loss(y[te], pred, tier_lc,
                                                       run_all.HARM_WEIGHTS)),
                 "conformal_cov": conformal_coverage(p_cal, y[cal], p, y[te])}
            gates = [r["ece_high"] <= 0.10, r["aurc_high"] <= 0.25, r["harm_loss"] <= 0.20,
                     r["conformal_cov"] >= 1 - ALPHA]
            r["gates_passed_of4"] = int(sum(gates))
            rows.append(r)
            if op == "clean":
                wrong = pred != y[te]
                n = len(wrong)
                for lad, (wl, wm, wh) in LADDERS.items():
                    w = {"low": wl, "medium": wm, "high": wh}
                    h = sum(w[t] * wrong[tier_lc == t].sum() for t in w) / n
                    wrows.append({"seed": seed, "model": name, "ladder": lad, "harm": h})
    return rows, wrows


def write(rows, name):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/" + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=42)
    args = ap.parse_args()
    run_all.N_PER_TIER = N_PER_TIER
    rows, wrows = [], []
    for k in range(args.seeds):
        seed = args.start + k
        np.random.seed(seed)
        run_all.SEED = seed
        X, y, _, tiers = run_all.build_cohort()
        r, w = run_seed(X, y, tiers, seed)
        rows += r
        wrows += w
        print(f"[seed {seed}] done", flush=True)
    write(rows, "operator_stress_seed.csv")

    clean = {(r["seed"], r["model"]): r for r in rows if r["operator"] == "clean"}
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model"], r["operator"], r["level"])
        c = clean[(r["seed"], r["model"])]
        for m in ("accuracy", "ece_high", "aurc_overall", "aurc_high", "harm_loss",
                  "conformal_cov", "gates_passed_of4"):
            agg[key][m].append(r[m])
        agg[key]["d_aurc"].append(r["aurc_overall"] - c["aurc_overall"])
        agg[key]["d_acc"].append(r["accuracy"] - c["accuracy"])
    summ = []
    for (model, op, lv), d in agg.items():
        row = {"model": model, "operator": op, "level": lv, "n_seeds": len(d["accuracy"])}
        for m, v in d.items():
            v = np.asarray(v, float)
            row[m + "_mean"] = round(float(v.mean()), 4)
            row[m + "_sd"] = round(float(v.std(ddof=1)), 4)
        row["undercover_seeds"] = int(sum(c < 1 - ALPHA for c in d["conformal_cov"]))
        summ.append(row)
    write(summ, "operator_stress_summary.csv")

    wagg = defaultdict(list)
    for r in wrows:
        wagg[(r["ladder"], r["model"])].append(r["harm"])
    out = []
    for lad in LADDERS:
        means = {m: float(np.mean(wagg[(lad, m)])) for m in FAMILIES}
        order = sorted(means, key=means.get)
        for m in FAMILIES:
            v = np.asarray(wagg[(lad, m)])
            out.append({"ladder": lad, "model": m, "harm_mean": round(means[m], 4),
                        "harm_sd": round(float(v.std(ddof=1)), 4),
                        "rank": order.index(m) + 1})
    write(out, "harm_weight_sensitivity.csv")


if __name__ == "__main__":
    main()
