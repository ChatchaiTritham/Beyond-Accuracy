"""Build results/synthetic_tier_guarantees_table.tex from results/synthetic_robustness_seed.csv."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results"
NAMES = {"logistic_regression": "LR", "random_forest": "RF-cal", "uncalibrated_forest": "RF-raw",
         "gradient_boosting": "GB", "xgboost": "XGBoost", "lightgbm": "LightGBM",
         "svm_rbf": "SVM-RBF", "mlp": "MLP", "mlp_calibrated": "MLP-Platt"}

d = pd.read_csv(OUT / "synthetic_robustness_seed.csv")
g = d.groupby("model")
lines = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Coverage, high tier} & \multicolumn{2}{c}{$|C|$, high tier}"
         r" & \multicolumn{3}{c}{Tier-wise LTT, high tier} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
         r"Family & marginal ($<$0.90) & Mondrian ($<$0.90) & marginal & Mondrian"
         r" & certified & held-out FNR & alert rate \\", r"\midrule"]
for k, lab in NAMES.items():
    x = d[d.model == k]
    cm, cn = x.cov_marginal_High, x.cov_mondrian_High
    cert = x.ltt_certified_High.mean()
    fnr = x.ltt_fnr_test_High.mean()
    al = x.ltt_alert_High.mean()
    tail = ("---", "---") if cert == 0 else (f"{fnr:.3f}", f"{al:.3f}")
    lines.append(f"{lab} & {cm.mean():.3f} ({int((cm < 0.9).sum())}) & {cn.mean():.3f} "
                 f"({int((cn < 0.9).sum())}) & {x.size_marginal_High.mean():.2f} & "
                 f"{x.size_mondrian_High.mean():.2f} & {int(round(cert * 20))}/20 & "
                 f"{tail[0]} & {tail[1]} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "synthetic_tier_guarantees_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
