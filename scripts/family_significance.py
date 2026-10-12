"""Seed-paired significance tests for the six-family sweep.

Reads results/model_seed_metrics.csv (written by scripts/model_sweep.py) and writes
results/family_significance.csv:
  * Friedman test across all model arms per metric (seeds are the blocks);
  * Wilcoxon signed-rank test, two-sided, for every pair of arms per metric,
    with Holm correction within each metric.
No resampling and no random state: the output is a deterministic function of the input.
"""
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "ece_overall", "ece_high", "aurc_overall", "harm_loss"]


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    df = pd.read_csv(ROOT / "results" / "model_seed_metrics.csv")
    rows = []
    for m in METRICS:
        wide = df.pivot(index="seed", columns="model", values=m).dropna()
        stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
        rows.append(dict(metric=m, test="friedman", a="all", b="all", n_seeds=len(wide),
                         statistic=stat, p=p, p_holm=p, median_diff=float("nan")))
        pairs = list(combinations(wide.columns, 2))
        res = [wilcoxon(wide[a], wide[b]) for a, b in pairs]
        adj = holm([r.pvalue for r in res])
        for (a, b), r, ph in zip(pairs, res, adj):
            rows.append(dict(metric=m, test="wilcoxon", a=a, b=b, n_seeds=len(wide),
                             statistic=r.statistic, p=r.pvalue, p_holm=ph,
                             median_diff=float((wide[a] - wide[b]).median())))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "family_significance.csv", index=False)
    print(out[out.test == "friedman"].to_string(index=False))


if __name__ == "__main__":
    main()
