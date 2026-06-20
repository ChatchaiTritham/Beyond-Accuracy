#!/usr/bin/env python3
"""
generate_figures.py  -  publication figures for the Beyond-Accuracy repository
------------------------------------------------------------------------------
"Beyond Accuracy: A Simulation-Based Governance Evaluation Framework"

This script renders the manuscript's figures into the repository's own
``figures/`` directory. Two kinds of figure are produced:

  Schematic diagrams (no asserted metric values):
    fig1-conceptual         - 3-pillar conceptual framework + TRI-X
    fig2-archetype-pipeline - archetype-to-scenario instantiation pipeline
    fig3-pipeline           - 6-stage evaluation pipeline
    fig6-v3-lifecycle       - V3 lifecycle positioning diagram
    fig-s1-lifecycle-detail - extended V3 lifecycle (supplementary)

  Data figures (rendered from results/, NOT hardcoded):
    fig4_coverage_risk      - coverage-risk curve, AURC from eval_results.json
    fig5_harm_xai           - tier-stratified harm + XAI stability from results/

Every numeric value in the data figures is read at run time from
``results/eval_results.json`` (written by ``run_all.py`` under seed 42).
Nothing is hardcoded, so the plots always track the repository's computed
outputs. To regenerate: run ``python run_all.py`` first, then this script.

Run:  python figures/generate_figures.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# Shared, byte-identical publication style/util (vendored from _management).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubviz import apply_pub_style, save_fig, PALETTE, load_results, results_dir  # noqa: E402

# -- Karger column dimensions (mm -> inches, 1 in = 25.4 mm) ------------------
SINGLE_COL_MM  = 80     # single column
ONEHALF_COL_MM = 120    # 1.5-column
DOUBLE_COL_MM  = 170    # double column (full width)

def mm2in(mm):
    return mm / 25.4

SINGLE_COL  = mm2in(SINGLE_COL_MM)    # ~3.15 in
ONEHALF_COL = mm2in(ONEHALF_COL_MM)   # ~4.72 in
DOUBLE_COL  = mm2in(DOUBLE_COL_MM)    # ~6.69 in

# -- Colour palette ----------------------------------------------------------
# All colours verified for WCAG AA contrast against white background
BD  = '#1C4E80'   # dibiBlue dark
BM  = '#2E7BBF'   # blue medium
BL  = '#D6E4F5'   # blue light tint
GD  = '#007040'   # dibiGreen dark
GM  = '#16A464'   # green medium
GL  = '#D1F0E0'   # green light tint
RD  = '#B41414'   # dibiRed dark
RL  = '#FAD9D9'   # red light tint
AM  = '#C06800'   # amber (dibiAmber)
AL  = '#FFF0D0'   # amber light tint
GRD = '#2D2D2D'   # near-black text
GRM = '#666666'   # grey medium
GRL = '#D4D4D4'   # grey light
WH  = '#FFFFFF'

# -- Paths: write into the repo figures/ dir; read from results/ -------------
HERE    = os.path.dirname(os.path.abspath(__file__))
REPO    = os.path.dirname(HERE)
OUTDIR  = HERE

# Canonical Top-Tier style (shared, vendored in pubviz). Data-figure series use
# the shared Okabe-Ito PALETTE; schematics keep branded box fills over the
# shared fonts/savefig settings (FIGURE_STYLE rule 7).
import matplotlib as mpl  # noqa: E402

apply_pub_style()
# Project-specific additions kept on top of the canonical base.
mpl.rcParams.update({
    'axes.unicode_minus': True,
    'xtick.direction': 'out', 'ytick.direction': 'out',
})

# Okabe-Ito aliases for the data-figure series (color-blind safe, palette order)
P_BLUE   = PALETTE[0]   # overall / low
P_ORANGE = PALETTE[1]   # medium
P_GREEN  = PALETTE[2]   # framework / accent
P_PINK   = PALETTE[3]   # high
P_AMBER  = PALETTE[4]


# ============================================================================
# Shared helpers
# ============================================================================

def _save(fig, name):
    """Save vector PDF + 300-dpi PNG (canonical) into the repo figures/ dir
    via the shared pubviz.save_fig helper (tight bbox, white bg, 300 dpi)."""
    save_fig(fig, name, out_dir=OUTDIR)


def _shadow_box(ax, cx, cy, w, h, offset=(0.04, -0.04),
                shadow_color='#00000012'):
    """Draw a subtle shadow behind a box (call before the main box)."""
    sx, sy = offset
    shadow = FancyBboxPatch(
        (cx - w / 2 + sx, cy - h / 2 + sy), w, h,
        boxstyle='round,pad=0.08',
        facecolor=shadow_color, edgecolor='none',
        linewidth=0, zorder=1, clip_on=False
    )
    ax.add_patch(shadow)


def _box(ax, cx, cy, w, h, lines,
         fc=BL, ec=BD, tc=GRD, fs=8.2, bold_first=False, lc=1.38,
         radius=0.08, shadow=True, lw=1.3):
    """Draw a rounded rectangle centred at (cx, cy) with size (w, h)."""
    if shadow:
        _shadow_box(ax, cx, cy, w, h)

    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f'round,pad={radius}',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=2, clip_on=False
    )
    ax.add_patch(patch)

    if bold_first and len(lines) > 1:
        n = len(lines)
        line_h = h / (n + 0.5)
        top_y  = cy + (n - 1) / 2 * line_h * 0.85
        for k, ln in enumerate(lines):
            fw = 'bold' if k == 0 else 'normal'
            ax.text(cx, top_y - k * line_h * 0.85, ln,
                    ha='center', va='center',
                    color=tc, fontsize=fs, fontweight=fw,
                    linespacing=lc, zorder=3, clip_on=False)
    else:
        ax.text(cx, cy, '\n'.join(lines),
                ha='center', va='center',
                color=tc, fontsize=fs, linespacing=lc,
                zorder=3, multialignment='center', clip_on=False)


def _arrow(ax, x0, y0, x1, y1, col=GRD, lw=1.5, ms=14):
    """Annotate-style arrow with minimum linewidth enforced."""
    lw = max(lw, 0.5)
    ax.annotate(
        '', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle='->', color=col,
                        lw=lw, mutation_scale=ms),
        annotation_clip=False, zorder=4
    )


def _badge(ax, cx, cy, r, num, bg=BD, fg=WH, fs=7.5):
    """Filled circle 'badge' with a number inside."""
    shadow = Circle((cx + 0.02, cy - 0.02), r,
                    facecolor='#00000015', edgecolor='none',
                    linewidth=0, zorder=5, clip_on=False)
    ax.add_patch(shadow)
    circle = Circle((cx, cy), r, facecolor=bg, edgecolor='#FFFFFF',
                    linewidth=0.8, zorder=6, clip_on=False)
    ax.add_patch(circle)
    ax.text(cx, cy, str(num), ha='center', va='center',
            fontsize=fs, fontweight='bold', color=fg, zorder=7,
            clip_on=False)


def _add_grid(ax, axis='y', alpha=0.25):
    """Add subtle reference grid to data plots."""
    ax.grid(axis=axis, linestyle='-', linewidth=0.3, alpha=alpha,
            color=GRM, zorder=0)
    ax.set_axisbelow(True)


# ============================================================================
# FIG 1 - Conceptual framework  (schematic; no metric values)
# ============================================================================
def make_fig1():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    fig.patch.set_facecolor(WH)

    for span, col, alpha in [((0.20, 3.50), BL, 0.07),
                              ((4.45, 7.30), BD, 0.04),
                              ((7.65, 10.00), GD, 0.05)]:
        ax.axvspan(span[0], span[1], ymin=0.05, ymax=0.90,
                   color=col, alpha=alpha, zorder=0)

    pillars = [
        (1.75, 3.42,
         ['Structured Clinical Reasoning', '(TiTrATE methodology)'],
         BL, BD),
        (1.75, 2.47,
         ['Urgency Stratification', '(triage tiers: L / M / H)'],
         GL, GD),
        (1.75, 1.52,
         ['Explainability-aware Design', '(XAI as audit artefact)'],
         AL, AM),
    ]
    for cx, cy, lines, fc, ec in pillars:
        _box(ax, cx, cy, 3.00, 0.72, lines, fc=fc, ec=ec,
             fs=8.0, lc=1.30, shadow=True)

    ax.text(1.75, 4.35, 'Domain Pillars',
            ha='center', va='center', fontsize=8.0,
            color=BD, style='italic', fontweight='bold')

    x_stub_end  = 3.60
    x_trix_left = 4.47

    for _, cy, *_ in pillars:
        _arrow(ax, 1.75 + 1.50, cy, x_stub_end, cy,
               col=GRM, lw=1.8, ms=12)

    ax.plot([x_stub_end, x_stub_end], [1.52, 3.42],
            color=GRM, lw=2.4, zorder=3, solid_capstyle='round')

    _arrow(ax, x_stub_end, 2.47, x_trix_left, 2.47,
           col=BD, lw=2.8, ms=18)

    _box(ax, 5.85, 2.47, 2.76, 1.90,
         ['TRI-X', 'Decision-Governance', 'Programme'],
         fc=BD, ec=BD, tc=WH, fs=11.0, bold_first=True,
         shadow=True, lw=1.5)

    ax.text(5.85, 1.78,
            'Uncertainty as Control Signal (escalation · abstention)',
            ha='center', va='center', fontsize=7.0, color='#C0D8F0',
            style='italic', zorder=5)

    ax.text(5.85, 4.35, 'Governance Paradigm',
            ha='center', va='center', fontsize=8.0,
            color=BD, style='italic', fontweight='bold')

    x_trix_right  = 7.23
    x_outbox_left = 7.76
    _arrow(ax, x_trix_right, 2.47, x_outbox_left, 2.47,
           col=GD, lw=2.8, ms=18)

    _box(ax, 8.85, 2.47, 2.18, 1.90,
         ['Simulation-based', 'Safety Evaluation', '(Beyond Accuracy)'],
         fc=GD, ec=GD, tc=WH, fs=9.2, bold_first=True,
         shadow=True, lw=1.5)

    ax.text(8.85, 4.35, 'Framework Output',
            ha='center', va='center', fontsize=8.0,
            color=GD, style='italic', fontweight='bold')

    ax.plot([0.20, 9.95], [4.18, 4.18],
            color=GRL, lw=0.7, zorder=1)

    _save(fig, 'fig1-conceptual')
    plt.close(fig)


# ============================================================================
# FIG 2 - Evaluation pipeline (schematic; no metric values)
# ============================================================================
def make_fig2():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.6))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 3.2)
    ax.axis('off')
    fig.patch.set_facecolor(WH)

    bw  = 1.40
    bh  = 1.10
    cy  = 1.75
    gap = 0.28
    x0  = 0.76
    step = bw + gap

    stages = [
        (x0 + 0 * step,
         ['Clinical', 'Archetypes', '+ Risk Tiers', '(L / M / H)'],
         BL, BD, GRD),

        (x0 + 1 * step,
         ['Scenario', 'Instantiation', '(mask · noise ·',
          'conflict · degrade)'],
         AL, AM, GRD),

        (x0 + 2 * step,
         ['System', 'Under Test', '(CDSS)'],
         '#EEF2F6', GRM, GRD),

        (x0 + 3 * step,
         ['Log Auditable', 'Artefacts', '(versioned;', 'seeded)'],
         BL, BD, GRD),

        (x0 + 4 * step,
         ['Compute Metrics',
          r'ECE, AURC,',
          r'$\mathcal{L}_\mathrm{harm}$, $\eta_\mathrm{xai}$,',
          r'$\hat{C}_\alpha$'],
         BL, BD, GRD),

        (x0 + 5 * step,
         ['Governance', 'Verdict', 'Pre-deployment', 'Evidence'],
         GD, GD, WH),
    ]

    badge_colours = [BD, AM, GRM, BD, BD, GD]

    for i, (cx, lines, fc, ec, tc) in enumerate(stages):
        _box(ax, cx, cy, bw, bh, lines, fc=fc, ec=ec, tc=tc,
             fs=7.2, lc=1.28, shadow=True)
        _badge(ax, cx, cy + bh / 2 + 0.20, 0.18,
               i + 1, bg=badge_colours[i])
        if i < len(stages) - 1:
            nx = stages[i + 1][0]
            _arrow(ax, cx + bw / 2 + 0.02, cy, nx - bw / 2 - 0.02, cy,
                   col=GRD, lw=2.0, ms=14)

    phase_labels = [
        (x0,             'Input',        BD),
        (x0 + step,      'Perturbation', AM),
        (x0 + 2 * step,  'Inference',    GRM),
        (x0 + 3 * step,  'Logging',      BD),
        (x0 + 4 * step,  'Metrics',      BD),
        (x0 + 5 * step,  'Verdict',      GD),
    ]
    for xc, label, col in phase_labels:
        ax.text(xc, 2.94, label, ha='center', fontsize=7.0,
                color=col, style='italic', fontweight='bold')

    _save(fig, 'fig3-pipeline')
    plt.close(fig)


# ============================================================================
# FIG 3 - Archetype-to-scenario pipeline (schematic; no metric values)
# ============================================================================
def make_fig3():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.4))
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 2.7)
    ax.axis('off')
    fig.patch.set_facecolor(WH)

    bw  = 1.72
    bh  = 1.10
    cy  = 1.42
    gap = 0.28
    step = bw + gap
    x0 = 0.92

    stages = [
        (x0 + 0 * step,
         ['Clinical Archetype',
          r'(abstract + tier',
          r'$r \in \{L,\,M,\,H\}$)'],
         BL, BD, GRD),

        (x0 + 1 * step,
         ['Perturbation', 'Operators',
          r'$p_\mathrm{mask}$, $p_\mathrm{noise}$,',
          r'$p_\mathrm{conflict}$, $p_\mathrm{degrade}$'],
         AL, AM, GRD),

        (x0 + 2 * step,
         ['Scenario Set',
          r'($N_a$ instances;',
          'varied uncertainty',
          r'profile $u_i$)'],
         GL, GD, GRD),

        (x0 + 3 * step,
         ['Safety Metrics',
          r'$\mathrm{ECE}_{(r)}$, AURC,',
          r'$\mathcal{L}_\mathrm{harm}$, $\eta_\mathrm{xai}$,',
          r'$\hat{C}_\alpha$'],
         GL, GD, GRD),
    ]

    badge_bg = [BD, AM, GD, GD]

    for i, (cx, lines, fc, ec, tc) in enumerate(stages):
        _box(ax, cx, cy, bw, bh, lines, fc=fc, ec=ec, tc=tc,
             fs=7.6, lc=1.28, shadow=True)
        _badge(ax, cx, cy + bh / 2 + 0.18, 0.17,
               i + 1, bg=badge_bg[i])
        if i < len(stages) - 1:
            nx = stages[i + 1][0]
            _arrow(ax, cx + bw / 2 + 0.02, cy, nx - bw / 2 - 0.02, cy,
                   col=GRD, lw=2.0, ms=14)

    last_cx    = x0 + 3 * step
    right_edge = last_cx + bw / 2
    assert right_edge < 8.8, f"Right edge {right_edge:.2f} exceeds xlim!"

    _save(fig, 'fig2-archetype-pipeline')
    plt.close(fig)


# ============================================================================
# FIG 4 - Coverage-risk curve  (data: results/eval_results.json)
#
# Plots the EMPIRICAL coverage-risk curves stored by run_all.py (overall + per
# tier), with a seeded-bootstrap 95% CI band around the overall curve. The AURC
# labels are the computed scalars. Selective vs overall FNR (from results) is
# annotated so the abstention trade-off is explicit.
# ============================================================================
def _xy(curve):
    """Sorted, NaN-dropped (coverage, risk) arrays from a stored curve dict."""
    cov = np.asarray(curve['coverage'], float)
    risk = np.array([np.nan if r is None else r for r in curve['risk']], float)
    m = ~np.isnan(risk)
    cov, risk = cov[m], risk[m]
    o = np.argsort(cov)
    return cov[o], risk[o]


def make_fig4(res):
    cr = res['coverage_risk']
    aurc_O = cr['aurc_overall']
    aurc_L = cr['aurc_by_tier']['Low']
    aurc_M = cr['aurc_by_tier']['Medium']
    aurc_H = cr['aurc_by_tier']['High']

    cov_O, risk_O = _xy(cr['curves']['overall'])
    cov_L, risk_L = _xy(cr['curves']['Low'])
    cov_M, risk_M = _xy(cr['curves']['Medium'])
    cov_H, risk_H = _xy(cr['curves']['High'])

    band = cr['bootstrap_band']
    b_cov = np.asarray(band['coverage'], float)
    b_lo = np.array([np.nan if v is None else v for v in band['risk_lower']], float)
    b_hi = np.array([np.nan if v is None else v for v in band['risk_upper']], float)
    bm = ~(np.isnan(b_lo) | np.isnan(b_hi))
    ci_pct = int(round(band['ci_level'] * 100))
    n_boot = band['n_bootstrap']

    op_cov = res['conformal']['empirical_coverage']
    op_risk = float(np.interp(op_cov, cov_O, risk_O))

    # selective vs overall FNR operating points (clarify the abstention trade)
    fnr_pts = res['coverage_risk']['fnr']['points']

    ymax = max(0.30, float(np.nanmax(risk_H)) * 1.15, float(np.nanmax(b_hi[bm])) * 1.10)

    fig, ax = plt.subplots(figsize=(ONEHALF_COL, 3.8))
    fig.patch.set_facecolor(WH)
    _add_grid(ax, axis='both', alpha=0.18)

    # 95% bootstrap CI band around the overall curve
    ax.fill_between(b_cov[bm], b_lo[bm], b_hi[bm], color=P_BLUE, alpha=0.15,
                    zorder=1, label=f'Overall {ci_pct}% CI ({n_boot} boot.)')

    # Per-tier empirical curves (color-blind-safe palette + distinct dashes)
    ax.plot(cov_L, risk_L, color=P_GREEN, lw=1.4, ls='--', alpha=0.90, zorder=3,
            label=f'Low urgency  (AURC = {aurc_L:.3f})')
    ax.plot(cov_M, risk_M, color=P_ORANGE, lw=1.4, ls='-.', alpha=0.90, zorder=3,
            label=f'Medium urgency  (AURC = {aurc_M:.3f})')
    ax.plot(cov_H, risk_H, color=P_PINK, lw=1.4, ls=(0, (1, 1.5)), alpha=0.90,
            zorder=3, label=f'High urgency  (AURC = {aurc_H:.3f})')
    ax.plot(cov_O, risk_O, color=P_BLUE, lw=2.4, zorder=4,
            label=f'Overall  (AURC = {aurc_O:.3f})')

    # operating marker at the empirical conformal coverage
    ax.axvline(op_cov, color=GRD, lw=0.9, ls=':', zorder=6)
    ax.scatter([op_cov], [op_risk], color=P_BLUE, s=50, zorder=8,
               clip_on=False, edgecolors=WH, linewidths=0.8)
    ax.annotate(
        f'conformal\ncov. = {op_cov:.3f}',
        xy=(op_cov, op_risk),
        xytext=(op_cov - 0.30, min(op_risk + 0.10, ymax * 0.85)),
        fontsize=7.5, color=GRD,
        arrowprops=dict(arrowstyle='->', color=GRD, lw=0.85),
        zorder=9
    )

    # Selective vs overall FNR callout (real, from results) -- clarifies that
    # the curve's "risk" is computed over RETAINED cases, while abstaining
    # positives shifts overall FNR. Pick the ~0.5-coverage operating point.
    p = min(fnr_pts, key=lambda d: abs(d['coverage'] - 0.5))
    ax.text(0.97, 0.04,
            f'At coverage {p["coverage"]:.2f}:  '
            f'selective FNR = {p["selective_fnr"]:.2f},  '
            f'overall FNR = {p["overall_fnr"]:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.6, color=GRD, style='italic')

    ax.set_xlabel('Coverage (fraction retained)', fontsize=10.0, labelpad=5)
    ax.set_ylabel(r'Selective risk among retained  $R(c)$',
                  fontsize=10.0, labelpad=5)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.01, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8.5)

    leg = ax.legend(loc='upper left', fontsize=7.0,
                    framealpha=0.95, edgecolor=GRL,
                    borderpad=0.65, labelspacing=0.40,
                    handlelength=2.2)
    leg.get_frame().set_linewidth(0.5)

    _save(fig, 'fig4_coverage_risk')
    plt.close(fig)


# ============================================================================
# FIG 5 - Harm-weighted loss | XAI stability  (data: results/eval_results.json)
#
# Panel A: tier-stratified harm-weighted loss, framework vs baseline, read from
#          harm_by_tier_framework / harm_by_tier_baseline. Overall reduction is
#          reduction_fraction. (Note: harm uses the repo tier weights 1/3/10,
#          so the High tier dominates the scale.)
# Panel B: explanation stability eta_xai (single computed value) plus the per-
#          tier calibration error (ECE) the governance gate flags. No fabricated
#          per-outcome-class breakdown is invented.
# ============================================================================
def make_fig5(res):
    harm = res['harm']
    fw = harm['harm_by_tier_framework']
    bl = harm['harm_by_tier_baseline']
    # Tiers ordered Low->Medium->High to match fig4's coverage-risk ordering.
    tiers   = ['Low', 'Medium', 'High']
    keys    = ['low', 'medium', 'high']
    fw_harm = [fw[k] for k in keys]
    bl_harm = [bl[k] for k in keys]
    # Seeded-bootstrap 95% CI -> asymmetric error bars on each bar.
    fw_ci = harm.get('harm_by_tier_framework_ci', {})
    bl_ci = harm.get('harm_by_tier_baseline_ci', {})

    def _err(vals, ci):
        lo = [max(0.0, v - ci[k]['lower']) for v, k in zip(vals, keys)] if ci else None
        hi = [max(0.0, ci[k]['upper'] - v) for v, k in zip(vals, keys)] if ci else None
        return np.array([lo, hi]) if ci else None

    fw_err = _err(fw_harm, fw_ci)
    bl_err = _err(bl_harm, bl_ci)
    reduction_pct = 100.0 * harm['reduction_fraction']
    fw_overall = harm['harm_weighted_loss_framework']
    bl_overall = harm['harm_weighted_loss_baseline']

    eta_xai = res['explanation_stability']['eta_xai_spearman']
    ece_tier = res['calibration']['ece_by_tier']
    ece_vals = [ece_tier['Low'], ece_tier['Medium'], ece_tier['High']]

    # constrained_layout (global) handles spacing; avoid manual GridSpec margins
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, 3.6),
        gridspec_kw=dict(width_ratios=[1.0, 0.82]))
    fig.patch.set_facecolor(WH)

    for ax in (ax_a, ax_b):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8.5)
        _add_grid(ax, axis='y', alpha=0.20)

    # -- PANEL A: tier-stratified harm loss --
    x     = np.arange(len(tiers))
    width = 0.36

    ebar = dict(ecolor=GRD, elinewidth=0.9, capsize=2.5, capthick=0.9)
    ax_a.bar(x - width / 2, bl_harm, width,
             color=GRL, edgecolor=GRM, linewidth=0.8,
             hatch='///', label='Accuracy-only baseline', zorder=2,
             yerr=bl_err, error_kw=ebar)
    ax_a.bar(x + width / 2, fw_harm, width,
             color=P_BLUE, edgecolor=P_BLUE, linewidth=0.8,
             label='Framework-guided', zorder=2,
             yerr=fw_err, error_kw=ebar)

    _hi_top = max([(v + (e if e else 0)) for v, e in
                   zip(bl_harm, (bl_err[1] if bl_err is not None else [0]*3))])
    ymax = _hi_top * 1.20
    for i, (b, f) in enumerate(zip(bl_harm, fw_harm)):
        if b > 0:
            pct = 100 * (b - f) / b
            ax_a.text(x[i] + width / 2, f + ymax * 0.02,
                      f'-{pct:.0f}%',
                      ha='center', va='bottom', fontsize=7.5,
                      color=GD, fontweight='bold')

    ax_a.text(0.97, 0.97,
              f'Overall harm-weighted loss\n'
              f'{bl_overall:.3f} -> {fw_overall:.3f}  (-{reduction_pct:.1f}%)',
              transform=ax_a.transAxes, ha='right', va='top',
              fontsize=7.0, color=GRD, style='italic', linespacing=1.30)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(tiers, fontsize=9.0)
    ax_a.set_xlabel('Urgency Tier', fontsize=10.0, labelpad=4)
    ax_a.set_ylabel(r'Harm-weighted Loss $\mathcal{L}_\mathrm{harm}$',
                    fontsize=10.0, labelpad=4)
    ax_a.set_ylim(0, ymax)
    ax_a.set_xlim(-0.55, 2.85)
    leg_a = ax_a.legend(loc='upper left', fontsize=7.0,
                        framealpha=0.95, edgecolor=GRL, borderpad=0.5)
    leg_a.get_frame().set_linewidth(0.5)
    ax_a.text(-0.08, 1.03, '(a)', transform=ax_a.transAxes,
              fontsize=11, fontweight='bold', va='bottom', ha='left')

    # -- PANEL B: per-tier calibration error + overall eta_xai --
    # palette + hatches so tiers stay distinct in greyscale (FIGURE_STYLE r.5)
    bar_col   = [P_BLUE, P_ORANGE, P_PINK]
    bar_hatch = ['\\\\', '...', '///']

    x2 = np.arange(len(tiers))
    bars = ax_b.bar(x2, ece_vals, 0.54,
                    color=bar_col, edgecolor=[c for c in bar_col],
                    linewidth=0.8, alpha=0.82, zorder=2)
    for bar, hatch in zip(bars, bar_hatch):
        bar.set_hatch(hatch)

    for xi, v in zip(x2, ece_vals):
        ax_b.text(xi, v + max(ece_vals) * 0.03, f'{v:.3f}',
                  ha='center', va='bottom', fontsize=8.0, color=GRD,
                  fontweight='bold')

    ax_b.axhline(eta_xai, color=P_GREEN, lw=1.0, ls='--', alpha=0.85, zorder=1,
                 label=fr'$\eta_\mathrm{{xai}}$ = {eta_xai:.3f} (overall)')

    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(tiers, fontsize=9.0)
    ax_b.set_xlabel('Urgency Tier', fontsize=10.0, labelpad=4)
    ax_b.set_ylabel(r'Calibration Error  ECE$(r)$',
                    fontsize=10.0, labelpad=4)
    ax_b.set_ylim(0, max(max(ece_vals), eta_xai) * 1.12)
    ax_b.set_xlim(-0.60, 2.60)

    leg_b = ax_b.legend(loc='upper left', fontsize=7.0,
                        framealpha=0.95, edgecolor=GRL, borderpad=0.5)
    leg_b.get_frame().set_linewidth(0.5)
    ax_b.text(-0.08, 1.03, '(b)', transform=ax_b.transAxes,
              fontsize=11, fontweight='bold', va='bottom', ha='left')

    _save(fig, 'fig5_harm_xai')
    plt.close(fig)


# ============================================================================
# Fig 6 - V3 Lifecycle Positioning (schematic; no metric values)
# ============================================================================
def make_fig6():
    print('[Fig 6] V3 Lifecycle Positioning ...')
    W = DOUBLE_COL
    H = W * 0.50
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    cy = 0.46
    bh = 0.30
    stages = [
        ('Verification',           0.08, cy, 0.22, bh, BL,  BD),
        ('Analytical\nValidation', 0.40, cy, 0.26, bh, GL,  GD),
        ('Clinical\nValidation',   0.74, cy, 0.22, bh, AL,  AM),
    ]

    for label, cx, _, bw, bh_, fc, ec in stages:
        x0, y0 = cx - bw/2, cy - bh_/2
        _shadow_box(ax, cx, cy, bw, bh_)
        box = FancyBboxPatch(
            (x0, y0), bw, bh_,
            boxstyle='round,pad=0.015',
            facecolor=fc, edgecolor=ec, linewidth=1.4,
            zorder=3
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.04, label, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=ec, zorder=4)

    arr_kw = dict(arrowstyle='->', lw=2.0, color=GRD, mutation_scale=14)
    ax.annotate('', xy=(0.40 - 0.26/2 - 0.02, cy),
                xytext=(0.08 + 0.22/2 + 0.02, cy),
                arrowprops=arr_kw, zorder=5)
    ax.annotate('', xy=(0.74 - 0.22/2 - 0.02, cy),
                xytext=(0.40 + 0.26/2 + 0.02, cy),
                arrowprops=arr_kw, zorder=5)

    zone_x = 0.40 - 0.26/2 - 0.03
    zone_w = 0.26 + 0.06
    sim_zone = FancyBboxPatch(
        (zone_x, 0.14), zone_w, 0.76,
        boxstyle='round,pad=0.02',
        facecolor=GD, alpha=0.06, edgecolor=GD,
        linewidth=1.0, linestyle='--', zorder=1
    )
    ax.add_patch(sim_zone)

    ax.text(0.40, 0.94,
            'Simulation-based evaluation zone  (no patient-level data required)',
            ha='center', va='top', fontsize=7.0, fontstyle='italic',
            color=GD, zorder=6)

    ax.annotate(
        'THIS FRAMEWORK',
        xy=(0.40, cy + bh/2 + 0.01),
        xytext=(0.40, 0.86),
        ha='center', va='center',
        fontsize=8.5, fontweight='bold', color=GD,
        arrowprops=dict(arrowstyle='->', color=GD, lw=1.8),
        zorder=6
    )

    ax.text(0.08, cy - 0.10, 'Sensor specs\nDevice testing',
            ha='center', va='center', fontsize=7.0, color=GRM, zorder=4)

    inner_items = [
        r'$\mathcal{P}$: 4 perturbation operators',
        r'$\mathcal{M}$: 5-metric safety suite',
        r'$\Phi$: governance verdict',
    ]
    for i, txt in enumerate(inner_items):
        ax.text(0.40, cy - 0.03 - i * 0.050, txt,
                ha='center', va='center', fontsize=7.0, color=GD,
                fontweight='medium', zorder=4)

    ax.text(0.74, cy - 0.10, 'Real patient data\nProspective trials',
            ha='center', va='center', fontsize=7.0, color=GRM, zorder=4)

    bot = cy - bh/2
    mid1 = (0.08 + 0.22/2 + 0.02 + 0.40 - 0.26/2 - 0.02) / 2
    ax.annotate('Archetype\nspec',
                xy=(mid1, bot - 0.01), xytext=(mid1, 0.10),
                ha='center', va='center', fontsize=7.0, color=BD,
                arrowprops=dict(arrowstyle='->', color=BD, lw=0.8,
                                linestyle='dashed'),
                zorder=5)

    mid2 = (0.40 + 0.26/2 + 0.02 + 0.74 - 0.22/2 - 0.02) / 2
    ax.annotate('Protocol\ncard',
                xy=(mid2, bot - 0.01), xytext=(mid2, 0.10),
                ha='center', va='center', fontsize=7.0, color=AM,
                arrowprops=dict(arrowstyle='->', color=AM, lw=0.8,
                                linestyle='dashed'),
                zorder=5)

    ax.text(0.50, 0.01,
            'V3 Digital Biomarker Development Lifecycle (Goldsack et al. 2020)',
            ha='center', va='bottom', fontsize=7.5, color=GRM,
            fontstyle='italic', zorder=6)

    _save(fig, 'fig6-v3-lifecycle')
    plt.close(fig)


# ============================================================================
# Fig S2 - Extended V3 Lifecycle (supplementary; schematic)
#
# The verdict band shows G = 0.60 (the computed governance score, Conditional)
# and the two governance gates the verdict function flags. These are pulled
# from results/ at run time, not hardcoded.
# ============================================================================
def make_fig_s2(res):
    print('[Fig S2] Extended V3 Lifecycle (supplementary) ...')
    gov = res['governance']
    score = gov['score']
    n_scenarios = res['meta']['cohort_n']

    W = DOUBLE_COL
    H = W * 0.92
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    arr_kw   = dict(arrowstyle='->', lw=1.6, color=GRD, mutation_scale=11)
    arr_thin = dict(arrowstyle='->', lw=1.0, color=GRD, mutation_scale=9)

    s1_cy = 0.893
    s1_bh = 0.092

    for label, cx, bw, fc, ec in [
        ('Verification',         0.10, 0.17, BL,  BD),
        ('Analytical\nValidation', 0.50, 0.26, GL,  GD),
        ('Clinical\nValidation', 0.89, 0.17, AL,  AM),
    ]:
        _shadow_box(ax, cx, s1_cy, bw, s1_bh, offset=(0.006, -0.006))
        ax.add_patch(FancyBboxPatch(
            (cx - bw/2, s1_cy - s1_bh/2), bw, s1_bh,
            boxstyle='round,pad=0.012', facecolor=fc, edgecolor=ec,
            linewidth=1.4, zorder=3))
        ax.text(cx, s1_cy, label, ha='center', va='center',
                fontsize=9.0, fontweight='bold', color=ec, zorder=4)

    ax.annotate('', xy=(0.50 - 0.13 - 0.018, s1_cy),
                xytext=(0.10 + 0.085 + 0.018, s1_cy),
                arrowprops=arr_kw, zorder=5)
    ax.annotate('', xy=(0.89 - 0.085 - 0.018, s1_cy),
                xytext=(0.50 + 0.13 + 0.018, s1_cy),
                arrowprops=arr_kw, zorder=5)

    ax.text(0.50, 0.975,
            'Simulation-based evaluation zone  (no patient-level data required)',
            ha='center', va='center', fontsize=7.0, fontstyle='italic',
            color=GD, zorder=6)
    ax.annotate('THIS FRAMEWORK',
                xy=(0.50, s1_cy + s1_bh/2 + 0.006),
                xytext=(0.50, 0.962),
                ha='center', va='center',
                fontsize=8.0, fontweight='bold', color=GD,
                arrowprops=dict(arrowstyle='->', color=GD, lw=1.8),
                zorder=6)

    ax.add_patch(FancyBboxPatch(
        (0.025, 0.085), 0.950, 0.756,
        boxstyle='round,pad=0.015',
        facecolor=GD, alpha=0.04, edgecolor=GD,
        linewidth=1.0, linestyle='--', zorder=1))

    def sbox(cx, cy, bw, bh, label, fc, ec, fs=7.5):
        ax.add_patch(FancyBboxPatch(
            (cx - bw/2, cy - bh/2), bw, bh,
            boxstyle='round,pad=0.010', facecolor=fc, edgecolor=ec,
            linewidth=1.0, zorder=3))
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fs, color=ec, zorder=4, linespacing=1.30)

    def down_arrow(cy_from, cy_to):
        ax.annotate('', xy=(0.50, cy_to), xytext=(0.50, cy_from),
                    arrowprops=dict(arrowstyle='->', lw=1.4,
                                   color=GRD, mutation_scale=10), zorder=5)

    ra_y, ra_h = 0.710, 0.090
    ax.text(0.50, ra_y + ra_h/2 + 0.026,
            'Input Pipeline',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=GD, zorder=4)

    row_a = [
        (0.12, r'$\mathcal{A}$: Archetype Set'
               '\n3 tiers (L/M/H)', BD, BL),
        (0.50, r'$\mathcal{P}$: Perturbation Operators'
               '\nmask · noise · conflict · degrade',  GD, GL),
        (0.88, r'$\mathcal{S}$: Scenario Instantiation'
               f'\n{n_scenarios} synthetic scenarios',  GD, GL),
    ]
    for cx, lbl, ec, fc in row_a:
        sbox(cx, ra_y, 0.23, ra_h, lbl, fc, ec, fs=7.5)

    ax.annotate('', xy=(0.50 - 0.115 - 0.014, ra_y),
                xytext=(0.12 + 0.115 + 0.014, ra_y),
                arrowprops=arr_thin, zorder=5)
    ax.annotate('', xy=(0.88 - 0.115 - 0.014, ra_y),
                xytext=(0.50 + 0.115 + 0.014, ra_y),
                arrowprops=arr_thin, zorder=5)

    down_arrow(ra_y - ra_h/2 - 0.002, ra_y - ra_h/2 - 0.032)

    rb_y, rb_h = 0.545, 0.088
    ax.text(0.50, rb_y + rb_h/2 + 0.026,
            r'$\mathcal{M}$: Safety Metric Suite  (5 metrics)',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=GD, zorder=4)

    metrics = [
        (0.09, 'ECE$(r)$\nCalibration'),
        (0.27, 'AURC\nCoverage-risk'),
        (0.50, r'$\mathcal{L}_\mathrm{harm}$' + '\nHarm-weighted'),
        (0.73, r'$\eta_\mathrm{xai}$' + '\nXAI stability'),
        (0.91, r'$\hat{C}_\alpha$' + '\nConf. coverage'),
    ]
    for cx, lbl in metrics:
        sbox(cx, rb_y, 0.155, rb_h, lbl, GL, GD, fs=7.0)

    down_arrow(rb_y - rb_h/2 - 0.002, rb_y - rb_h/2 - 0.030)

    rc_y, rc_h = 0.375, 0.090
    ax.text(0.50, rc_y + rc_h/2 + 0.026,
            r'$\Phi$: Verdict Function  (5-gate, institution-negotiable thresholds)',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=AM, zorder=4)

    verdicts = [
        (0.17, BD, BL,  'Approve\n$G = 1.0$'),
        (0.50, AM, AL,  f'Conditional\n$G = {score:.2f}$\n[this run]'),
        (0.83, RD, RL,  'Reject\n$G \\leq 0.40$'),
    ]
    for cx, ec, fc, lbl in verdicts:
        sbox(cx, rc_y, 0.28, rc_h, lbl, fc, ec, fs=7.5)

    failed = ', '.join(gov.get('failed_gates', []))
    ax.text(0.50, rc_y - rc_h/2 - 0.008,
            f'{gov["n_passed"]}/5 gates pass; failed: {failed}',
            ha='center', va='top', fontsize=7.0, color=AM,
            fontstyle='italic', zorder=4)

    down_arrow(rc_y - rc_h/2 - 0.036, rc_y - rc_h/2 - 0.062)

    rd_y, rd_h = 0.175, 0.075
    ax.text(0.50, rd_y + rd_h/2 + 0.026,
            'Governance Artefacts',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=GRD, zorder=4)

    artefacts = [
        (0.12, 'Protocol\nCard',              BD, BL),
        (0.37, 'Coverage-Risk\nCurve',        GD, GL),
        (0.63, 'XAI Stability\nReport',       GD, GL),
        (0.88, 'Conformal\nBound $\\hat{C}_{0.10}$', BM, BL),
    ]
    for cx, lbl, ec, fc in artefacts:
        sbox(cx, rd_y, 0.21, rd_h, lbl, fc, ec, fs=7.0)

    bot  = s1_cy - s1_bh/2
    mid1 = (0.10 + 0.085 + 0.015 + 0.50 - 0.13 - 0.015) / 2
    mid2 = (0.50 + 0.13 + 0.015 + 0.89 - 0.085 - 0.015) / 2
    lbl_y = bot - 0.054

    ax.annotate('Archetype spec',
                xy=(mid1, bot - 0.004), xytext=(mid1, lbl_y),
                ha='center', va='top', fontsize=7.0, color=BD,
                arrowprops=dict(arrowstyle='->', color=BD, lw=0.9,
                                linestyle='dashed'), zorder=6)
    ax.annotate('Protocol card',
                xy=(mid2, bot - 0.004), xytext=(mid2, lbl_y),
                ha='center', va='top', fontsize=7.0, color=AM,
                arrowprops=dict(arrowstyle='->', color=AM, lw=0.9,
                                linestyle='dashed'), zorder=6)

    ax.add_patch(FancyBboxPatch(
        (0.025, 0.016), 0.950, 0.054,
        boxstyle='round,pad=0.008',
        facecolor=BD, alpha=0.07, edgecolor=BD,
        linewidth=0.8, zorder=2))
    ax.text(0.50, 0.043,
            'Regulatory alignment:  DECIDE-AI (Stages 1-2)  ·  '
            'EU AI Act Art. 9, 13, 14  ·  '
            'FDA GMLP Principle 6  ·  ISO 14971 Cl. 5',
            ha='center', va='center', fontsize=7.0, color=BD, zorder=4)

    _save(fig, 'fig-s1-lifecycle-detail')
    plt.close(fig)


# ============================================================================
# FIG 7 - Decision-curve analysis  (data: results/eval_results.json)
#
# Net benefit vs threshold probability for the model, treat-all and treat-none
# reference strategies. Computed in run_all.py directly from the held-out fold
# (seed 42); nothing is hardcoded. This is the decision-analytic figure the
# "beyond accuracy" framing requires: it shows the clinical value of acting on
# the model's probabilities across plausible treatment thresholds.
# ============================================================================
def make_fig_dca(res):
    dca = res['decision_curve']
    t = np.asarray(dca['thresholds'], float)
    nb_model = np.asarray(dca['net_benefit_model'], float)
    nb_all = np.asarray(dca['net_benefit_treat_all'], float)
    prev = dca['prevalence']

    fig, ax = plt.subplots(figsize=(ONEHALF_COL, 3.6))
    fig.patch.set_facecolor(WH)
    _add_grid(ax, axis='both', alpha=0.18)

    ax.axhline(0.0, color=GRM, lw=1.0, ls='-', alpha=0.7, zorder=2,
               label='Treat none')
    ax.plot(t, nb_all, color=P_ORANGE, lw=1.5, ls='--', zorder=3,
            label='Treat all')
    ax.plot(t, nb_model, color=P_BLUE, lw=2.4, zorder=4,
            label='Framework (model-guided)')

    # mark the framework operating threshold for the high-urgency tier (0.30)
    op_t = res['meta']['tier_threshold']['high']
    op_nb = float(np.interp(op_t, t, nb_model))
    ax.axvline(op_t, color=GRD, lw=0.9, ls=':', zorder=5)
    ax.scatter([op_t], [op_nb], color=P_BLUE, s=45, zorder=8,
               edgecolors=WH, linewidths=0.8, clip_on=False)
    ax.annotate(
        f'high-urgency\nthreshold = {op_t:.2f}\nNB = {op_nb:.3f}',
        xy=(op_t, op_nb), xytext=(op_t + 0.07, op_nb - 0.14),
        fontsize=7.0, color=GRD,
        arrowprops=dict(arrowstyle='->', color=GRD, lw=0.85), zorder=9)

    ax.set_xlabel('Threshold probability  $p_t$', fontsize=10.0, labelpad=5)
    ax.set_ylabel('Net benefit', fontsize=10.0, labelpad=5)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(min(-0.02, float(nb_all.min()) * 1.05),
                float(max(nb_model.max(), prev)) * 1.12)
    ax.tick_params(labelsize=8.5)
    leg = ax.legend(loc='lower left', fontsize=7.5,
                    framealpha=0.95, edgecolor=GRL, borderpad=0.6)
    leg.get_frame().set_linewidth(0.5)

    _save(fig, 'fig7_decision_curve')
    plt.close(fig)


# ============================================================================
# FIG 8 - Reliability diagram  (data: results/eval_results.json)
#
# Predicted probability vs observed frequency by bin, with the computed ECE
# annotated. Bins and ECE are produced by run_all.py from the repo's own
# reliability_curve()/ECE, so the diagram and the reported ECE are consistent.
# ============================================================================
def make_fig_reliability(res):
    rel = res['calibration']['reliability_overall']
    conf = np.asarray(rel['bin_confidence'], float)
    acc = np.asarray(rel['bin_accuracy'], float)
    cnt = np.asarray(rel['bin_count'], float)
    ece = rel['ece']

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.45, 3.6))
    fig.patch.set_facecolor(WH)
    _add_grid(ax, axis='both', alpha=0.18)

    # perfect-calibration diagonal
    ax.plot([0, 1], [0, 1], color=GRM, lw=1.0, ls='--', zorder=2,
            label='Perfect calibration')

    # gap bars (predicted - observed) as light shading under the curve markers
    for c, a in zip(conf, acc):
        ax.plot([c, c], [c, a], color=RD, lw=1.0, alpha=0.45, zorder=3)

    # marker size scaled by bin count (more mass = larger), real counts
    smax = cnt.max() if cnt.size else 1.0
    sizes = 30 + 120 * (cnt / smax)
    ax.plot(conf, acc, color=P_BLUE, lw=1.8, zorder=4,
            label='Observed (per bin)')
    ax.scatter(conf, acc, s=sizes, color=P_BLUE, edgecolors=WH,
               linewidths=0.8, zorder=5)

    ax.text(0.04, 0.96, f'ECE = {ece:.3f}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9.0, color=GRD, fontweight='bold')
    ax.text(0.04, 0.88, 'marker area $\\propto$ bin count;\n'
            'red stems = calibration gap',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=6.6, color=GRM, style='italic')

    ax.set_xlabel('Mean predicted probability', fontsize=10.0, labelpad=5)
    ax.set_ylabel('Observed frequency', fontsize=10.0, labelpad=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=8.5)
    leg = ax.legend(loc='lower right', fontsize=7.5,
                    framealpha=0.95, edgecolor=GRL, borderpad=0.6)
    leg.get_frame().set_linewidth(0.5)

    _save(fig, 'fig8_reliability')
    plt.close(fig)


# ============================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('  Beyond Accuracy - Figure Generation')
    print('  Data figures rendered from results/eval_results.json')
    print('=' * 60)
    print()

    res = load_results('eval_results.json')
    os.makedirs(OUTDIR, exist_ok=True)

    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4(res)
    make_fig5(res)
    make_fig6()
    make_fig_s2(res)
    make_fig_dca(res)
    make_fig_reliability(res)

    print()
    print('=' * 60)
    print('  All 9 figures generated (PDF + PNG)')
    print(f'  Output: {OUTDIR}')
    print('  Data panels (fig4, fig5) and the fig-s1 verdict band')
    print('  are rendered from results/, not hardcoded.')
    print('=' * 60)
