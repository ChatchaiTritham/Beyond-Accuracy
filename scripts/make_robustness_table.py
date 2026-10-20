"""Build results/ktas_robustness_table.tex from results/ktas_robustness_seed.csv."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results"
NAMES = {"logistic_regression": "Logistic reg.", "random_forest": "RF (isotonic)",
         "gradient_boosting": "Grad. boosting", "mlp_calibrated": "MLP (Platt)",
         "xgboost": "XGBoost", "lightgbm": "LightGBM"}
COLS = ["fnr_high_single05", "fnr_high_expert_policy", "fnr_high_nurse_policy",
        "harm_nurse_policy", "cov_high_marginal", "cov_high_mondrian_nurse",
        "size_high_marginal", "size_high_mondrian_nurse"]

d = pd.read_csv(OUT / "ktas_robustness_seed.csv")
m, s = d.groupby("model")[COLS].mean(), d.groupby("model")[COLS].std()
lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{FNR, KTAS 1--2} & & \multicolumn{2}{c}{Coverage, KTAS 1--2}"
         r" & \multicolumn{2}{c}{$|C|$, KTAS 1--2} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Family & $\tau{=}0.5$ & expert tier & nurse tier & Harm, nurse & marginal & Mondrian"
         r" & marginal & Mondrian \\", r"\midrule"]
for k, lab in NAMES.items():
    v = [f"{m.loc[k, c]:.3f}" for c in COLS]
    v[2] = f"{m.loc[k, COLS[2]]:.3f} $\\pm$ {s.loc[k, COLS[2]]:.3f}"
    lines.append(lab + " & " + " & ".join(v) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "ktas_robustness_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
