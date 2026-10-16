"""Render the KTAS case-study table (LaTeX) from results/ktas_summary.csv.

Run after scripts/ktas_case_study.py:  python scripts/make_ktas_table.py
Writes results/ktas_table.tex; the manuscript \\input's this file unchanged.
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
if not (RES / "ktas_summary.csv").exists():
    RES = Path(__file__).resolve().parents[1] / "results_ktas"
s = pd.read_csv(RES / "ktas_summary.csv").set_index("model")
ORDER = [("logistic_regression", "Logistic reg."),
         ("random_forest", "RF (isotonic)"),
         ("gradient_boosting", "Grad. boosting"),
         ("mlp_calibrated", "MLP (Platt)")]


def ms(row, k, d=3):
    return f"{row[k + '_mean']:.{d}f} $\\pm$ {row[k + '_std']:.{d}f}"


def pct(v):
    return f"{100 * v:.0f}"


lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         " & " + " & ".join(n for _, n in ORDER) + r" \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{Standard audit}} \\"]
rows = [("AUROC", lambda r: ms(r, "auroc")),
        ("Brier score", lambda r: ms(r, "brier")),
        ("ECE, pooled", lambda r: ms(r, "ece_overall")),
        ("Passes audit (\\%)", lambda r: pct(r["standard_audit_pass_rate"])),
        (None, r"\multicolumn{5}{@{}l}{\emph{Tier-stratified suite}} \\"),
        ("ECE, KTAS 4--5", lambda r: ms(r, "ece_Low")),
        ("ECE, KTAS 3", lambda r: ms(r, "ece_Medium")),
        ("ECE, KTAS 1--2", lambda r: ms(r, "ece_High")),
        ("AURC, KTAS 1--2", lambda r: ms(r, "aurc_High")),
        ("FNR KTAS 1--2, $\\tau{=}0.5$", lambda r: ms(r, "fnr_high_baseline")),
        ("FNR KTAS 1--2, tier $\\tau$", lambda r: ms(r, "fnr_high_framework")),
        ("Harm loss, $\\tau{=}0.5$", lambda r: ms(r, "harm_baseline")),
        ("Harm loss, tier $\\tau$", lambda r: ms(r, "harm_framework")),
        ("Conformal coverage", lambda r: ms(r, "conformal_coverage")),
        ("Explanation stability", lambda r: ms(r, "eta_occlusion")),
        ("LTT certified (\\%)", lambda r: pct(r["ltt_certified_rate"])),
        ("Score $G$", lambda r: ms(r, "G", 2)),
        ("Conditional / Reject (\\%)",
         lambda r: f"{pct(r['verdict_Conditional_rate'])} / {pct(r['verdict_Reject_rate'])}")]
for label, f in rows:
    if label is None:
        lines += [r"\midrule", f]
        continue
    lines.append(label + " & " + " & ".join(f(s.loc[m]) for m, _ in ORDER) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(RES / "ktas_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
