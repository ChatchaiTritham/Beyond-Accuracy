"""Generate the cross-site LaTeX table from results/ktas_cross_site_summary.csv."""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
s = pd.read_csv(R / "ktas_cross_site_summary.csv")
NAMES = {"logistic_regression": "Logistic reg.", "random_forest": "RF (isotonic)",
         "gradient_boosting": "Grad. boosting", "mlp_calibrated": "MLP (Platt)"}
DIRS = {"site1->site2": "Develop on site 1, test on site 2 ($n{=}579$)",
        "site2->site1": "Develop on site 2, test on site 1 ($n{=}688$)"}
f = lambda r, k: f"{r[k + '_mean']:.3f}"
out = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
       r" & & \multicolumn{2}{c}{ECE} & & \multicolumn{3}{c}{Conformal ($\alpha{=}0.10$)} \\",
       r"\cmidrule(lr){3-4}\cmidrule(lr){6-8}",
       r"Family & AUROC & pooled & KTAS 1--2 & FNR$_{1\text{--}2}$ & unweighted ($<$0.90) & est.\ $w$ & $|C|$, est.\ $w$ \\",
       r"\midrule"]
for d, lab in DIRS.items():
    out.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{lab}}}}} \\")
    for m, nm in NAMES.items():
        r = s[(s.direction == d) & (s.model == m)].iloc[0]
        out.append(f"{nm} & {f(r,'auroc')} & {f(r,'ece_overall')} & {f(r,'ece_High')} & "
                   f"{f(r,'fnr_high_tier')} & {f(r,'cov_unweighted')} ({int(r.seeds_unweighted_below_090)}) & "
                   f"{f(r,'cov_weighted')} & {r.size_weighted_mean:.2f} \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
(R / "ktas_external_table.tex").write_text("\n".join(out) + "\n")
print("\n".join(out))
