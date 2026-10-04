"""Figures for the two guarantee experiments (scripts/shift_experiments.py).

fig_shift_coverage.pdf : coverage and prediction-set size against shift strength, for the
                         unweighted, oracle-weighted and estimated-weighted arms.
fig_risk_control.pdf   : the Learn-then-Test certificate -- empirical FNR and p-value per
                         candidate threshold, and the alert burden the certified threshold
                         implies.

Usage: python scripts/make_guarantee_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

STYLE = {
    "unweighted": ("#0072B2", (0, ()), "o"),
    "oracle_weighted": ("#D55E00", (0, (6, 2)), "s"),
    "estimated_weighted": ("#009E73", (0, (2, 1.5)), "^"),
}
LABEL = {"unweighted": "Unweighted split conformal",
         "oracle_weighted": "Weighted, oracle ratio",
         "estimated_weighted": "Weighted, estimated ratio"}


def fig_coverage() -> None:
    d = pd.read_csv(RES / "shift_coverage.csv")
    target = 1 - float(d["alpha"].iloc[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.6))
    for arm, g in d.groupby("weighting"):
        g = g.sort_values("tilt")
        col, dash, mk = STYLE[arm]
        a.plot(g.tilt, g.coverage, color=col, linestyle=dash, marker=mk, markersize=3.4,
               linewidth=1.3, markerfacecolor="white", label=LABEL[arm])
        b.plot(g.tilt, g.avg_set_size, color=col, linestyle=dash, marker=mk, markersize=3.4,
               linewidth=1.3, markerfacecolor="white")
    a.axhline(target, color="#4D4D4D", linestyle=(0, (1, 2)), linewidth=1.0)
    a.annotate(f"target {target:.2f}", (d.tilt.min(), target), xytext=(2, -10),
               textcoords="offset points", fontsize=7, color="#4D4D4D")
    a.set_xlabel(r"Shift strength $\lambda_{\mathrm{tilt}}$")
    a.set_ylabel("Empirical coverage")
    a.set_ylim(0.85, 1.01)
    b.set_xlabel(r"Shift strength $\lambda_{\mathrm{tilt}}$")
    b.set_ylabel(r"Mean prediction-set size $|C(x)|$")
    b.axhline(1.0, color="#999999", linewidth=0.6)
    a.text(0.0, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    b.text(0.0, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(OUT / "fig_shift_coverage.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_shift_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_shift_coverage.pdf")


def fig_risk() -> None:
    d = pd.read_csv(RES / "risk_control.csv").sort_values("lambda")
    v = pd.read_csv(RES / "risk_control_validation.csv").iloc[0]
    alpha_risk, delta = float(d.alpha_risk.iloc[0]), float(d.delta.iloc[0])

    fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.6))
    a.plot(d["lambda"], d.empirical_fnr_cal, color="#0072B2", marker="o", markersize=3.2,
           linewidth=1.3, markerfacecolor="white", label="FNR (calibration)")
    a.plot(d["lambda"], d.fnr_test_at_lambda, color="#D55E00", linestyle=(0, (6, 2)),
           marker="s", markersize=3.2, linewidth=1.3, markerfacecolor="white",
           label="FNR (held out)")
    a.axhline(alpha_risk, color="#4D4D4D", linestyle=(0, (1, 2)), linewidth=1.0)
    a.annotate(rf"$\alpha_{{\mathrm{{risk}}}}={alpha_risk:g}$",
               (d["lambda"].max(), alpha_risk), xytext=(-4, 4), textcoords="offset points",
               ha="right", fontsize=7, color="#4D4D4D")
    cert = d[d.certified]
    if len(cert):
        a.scatter(cert["lambda"], cert.empirical_fnr_cal, s=60, facecolors="none",
                  edgecolors="#009E73", linewidths=1.3, zorder=5, label="certified by LTT")
    a.set_xlabel(r"Alert threshold $\lambda$")
    a.set_ylabel("False-negative rate, high-urgency tier")
    a.legend(frameon=False, loc="upper left")

    b.plot(d["lambda"], d.alert_rate_high_tier, color="#CC79A7", marker="D", markersize=3.0,
           linewidth=1.3, markerfacecolor="white")
    if len(cert):
        b.axvline(float(cert["lambda"].max()), color="#009E73", linestyle=(0, (4, 2)),
                  linewidth=1.0)
        b.annotate("certified
operating point",
                   (float(cert["lambda"].max()), 0.18), xytext=(10, 0),
                   textcoords="offset points", fontsize=7, color="#009E73",
                   linespacing=1.2)
    b.set_xlabel(r"Alert threshold $\lambda$")
    b.set_ylabel("Alert rate, high-urgency tier")
    b.set_ylim(0, 1.02)
    a.text(0.0, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    b.text(0.0, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    fig.suptitle("")
    fig.text(0.5, -0.06, f"Validation over {int(v.n_deployments)} simulated deployments: "
                         f"violation rate {v.violation_rate:.3f} "
                         rf"$\leq\delta={delta:g}$", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_risk_control.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_risk_control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_risk_control.pdf")


if __name__ == "__main__":
    fig_coverage()
    fig_risk()
