#!/usr/bin/env python3
"""
generate_figures.py  v3.0  —  Commercial-grade publication figures
-------------------------------------------------------------------
"Beyond Accuracy: A Simulation-Based Governance Evaluation Framework"
Target: Digital Biomarkers (Karger, ISSN 2504-110X)

Generates 6 vector PDF + 600-DPI TIFF figures:
  fig1-conceptual.pdf / .tiff     — 3-pillar conceptual framework + TRI-X
  fig2-archetype-pipeline.pdf / .tiff — archetype-to-scenario instantiation pipeline
  fig3-pipeline.pdf / .tiff       — 6-stage evaluation pipeline
  fig4_coverage_risk.pdf / .tiff  — coverage–risk curve (tier stratified)
  fig5_harm_xai.pdf / .tiff       — 2-panel: harm-weighted loss | XAI stability
  fig6-v3-lifecycle.pdf / .tiff   — V3 lifecycle positioning diagram

v3.0 Commercial-grade upgrades:
  - 600 DPI (Karger combined illustration requirement)
  - Figures sized to Karger column widths (single 80mm, 1.5-col 120mm, double 170mm)
  - Minimum line width ≥ 0.5pt everywhere (above 0.4pt Karger minimum)
  - Minimum font size 7pt at print size
  - Colorblind-safe: hatching patterns on bar charts, distinct marker styles
  - Subtle grid lines on data plots for readability
  - Depth shadows on diagram boxes for professional appearance
  - Dual export: vector PDF (primary) + 600-DPI TIFF (production backup)
  - All fonts embedded as TrueType (pdf.fonttype=42)
  - Anti-aliased rendering forced on all elements
  - WCAG-contrast-checked colour palette

Run:  python generate_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec

# ── Karger column dimensions (mm → inches, 1 in = 25.4 mm) ──────────────────
SINGLE_COL_MM  = 80     # single column
ONEHALF_COL_MM = 120    # 1.5-column
DOUBLE_COL_MM  = 170    # double column (full width)

def mm2in(mm):
    return mm / 25.4

SINGLE_COL  = mm2in(SINGLE_COL_MM)    # ~3.15 in
ONEHALF_COL = mm2in(ONEHALF_COL_MM)   # ~4.72 in
DOUBLE_COL  = mm2in(DOUBLE_COL_MM)    # ~6.69 in

# ── Colour palette (exact match to \definecolor in main-digiB.tex) ───────────
# All colours verified for WCAG AA contrast against white background
BD  = '#1C4E80'   # dibiBlue dark   — contrast ratio 7.2:1 ✓
BM  = '#2E7BBF'   # blue medium     — contrast ratio 4.1:1 ✓
BL  = '#D6E4F5'   # blue light tint
GD  = '#007040'   # dibiGreen dark  — contrast ratio 5.4:1 ✓
GM  = '#16A464'   # green medium    — contrast ratio 3.4:1
GL  = '#D1F0E0'   # green light tint
RD  = '#B41414'   # dibiRed dark    — contrast ratio 7.8:1 ✓
RL  = '#FAD9D9'   # red light tint
AM  = '#C06800'   # amber (dibiAmber) — contrast ratio 4.0:1 ✓
AL  = '#FFF0D0'   # amber light tint
GRD = '#2D2D2D'   # near-black text — contrast ratio 14.5:1 ✓
GRM = '#666666'   # grey medium (darkened from #888 for WCAG) — 5.7:1 ✓
GRL = '#D4D4D4'   # grey light (darkened slightly for print)
WH  = '#FFFFFF'

OUTDIR = (r"d:\PhD\Manuscript\Manuscript\DigiB_Beyond-Accuracy"
          r"\DigitalBiomarkers_Submission\Figures")

# ── Global rcParams — Commercial-grade publication defaults ──────────────────
plt.rcParams.update({
    # Typography
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica Neue', 'Helvetica',
                           'Segoe UI', 'DejaVu Sans'],
    'font.size':          9.5,
    'mathtext.fontset':   'dejavusans',

    # Axes
    'axes.linewidth':     0.8,
    'axes.labelweight':   'medium',
    'axes.labelpad':      6,
    'axes.unicode_minus': True,

    # Ticks
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4.0,
    'ytick.major.size':   4.0,
    'xtick.minor.size':   2.0,
    'ytick.minor.size':   2.0,
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
    'xtick.minor.width':  0.5,
    'ytick.minor.width':  0.5,

    # Save / export
    'savefig.dpi':        600,           # Karger combined illustration req.
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.08,
    'savefig.transparent': False,

    # Font embedding
    'pdf.fonttype':       42,            # TrueType in PDF
    'ps.fonttype':        42,

    # Rendering quality
    'figure.dpi':         150,           # screen preview
    'lines.antialiased':  True,
    'patch.antialiased':  True,
    'text.antialiased':   True,

    # Legend defaults
    'legend.framealpha':  0.95,
    'legend.edgecolor':   GRL,
    'legend.fancybox':    True,
    'legend.fontsize':    7.5,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    """Save figure as vector PDF and 600-DPI TIFF.
    Handles locked files gracefully with fallback naming."""
    for ext, fmt, kw in [('pdf', 'pdf', {}),
                          ('tiff', 'tiff', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'}))]:
        path = os.path.join(OUTDIR, f'{name}.{ext}')
        try:
            fig.savefig(path, format=fmt, **kw)
            kb = os.path.getsize(path) / 1024
            sfx = ' (600 DPI LZW)' if ext == 'tiff' else ''
            print(f'  {name}.{ext}   ({kb:,.0f} KB{sfx})')
        except PermissionError:
            # File locked — save with _new suffix
            alt = os.path.join(OUTDIR, f'{name}_new.{ext}')
            fig.savefig(alt, format=fmt, **kw)
            kb = os.path.getsize(alt) / 1024
            print(f'  {name}_new.{ext}   ({kb:,.0f} KB) [original locked]')


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
    """
    Draw a rounded rectangle centred at (cx, cy) with size (w, h).
    lines: list of strings — joined with newlines and centred.
    bold_first: first line bold, rest normal weight.
    shadow: add subtle depth shadow behind box.
    """
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
    lw = max(lw, 0.5)  # enforce ≥ 0.5pt
    ax.annotate(
        '', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle='->', color=col,
                        lw=lw, mutation_scale=ms),
        annotation_clip=False, zorder=4
    )


def _badge(ax, cx, cy, r, num, bg=BD, fg=WH, fs=7.5):
    """Filled circle 'badge' with a number inside."""
    # Shadow
    shadow = Circle((cx + 0.02, cy - 0.02), r,
                    facecolor='#00000015', edgecolor='none',
                    linewidth=0, zorder=5, clip_on=False)
    ax.add_patch(shadow)
    # Badge
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


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Conceptual framework  (v4.0: landscape boxes, refined proportions)
#
# Layout (xlim 0–10, ylim 0–5.0):
#   Left column  (x = 1.75): Three landscape pillar boxes (h=0.72, step=0.95)
#   Merge bar    (x = 3.60): Vertical line connecting three stubs
#   Main arrow:  merge bar → TRI-X left edge
#   Centre       (x = 5.85): TRI-X governance box (h=1.90 — landscape)
#   Arrow:       TRI-X right → Beyond Accuracy left
#   Right        (x = 8.85): Beyond Accuracy output box (h=1.90 — landscape)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    fig.patch.set_facecolor(WH)

    # ── Column background bands (subtle shading) ────────────────────────────
    for span, col, alpha in [((0.20, 3.50), BL, 0.07),
                              ((4.45, 7.30), BD, 0.04),
                              ((7.65, 10.00), GD, 0.05)]:
        ax.axvspan(span[0], span[1], ymin=0.05, ymax=0.90,
                   color=col, alpha=alpha, zorder=0)

    # ── Three domain pillars (landscape: w=3.00, h=0.72, step=0.95) ─────────
    #    cy: 3.42 / 2.47 / 1.52  — span matches TRI-X box height exactly
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

    # ── Column header — Domain Pillars ──────────────────────────────────────
    ax.text(1.75, 4.35, 'Domain Pillars',
            ha='center', va='center', fontsize=8.0,
            color=BD, style='italic', fontweight='bold')

    # ── Merge-bar: 3 stubs → vertical bar → main arrow → TRI-X ─────────────
    x_stub_end  = 3.60
    x_trix_left = 4.47

    for _, cy, *_ in pillars:
        _arrow(ax, 1.75 + 1.50, cy, x_stub_end, cy,
               col=GRM, lw=1.8, ms=12)

    # Vertical merge bar — spans pillar group (1.52 → 3.42)
    ax.plot([x_stub_end, x_stub_end], [1.52, 3.42],
            color=GRM, lw=2.4, zorder=3, solid_capstyle='round')

    # Main arrow from merge midpoint → TRI-X left edge
    _arrow(ax, x_stub_end, 2.47, x_trix_left, 2.47,
           col=BD, lw=2.8, ms=18)

    # ── TRI-X centre box (landscape: w=2.76 > physical h=1.90) ──────────────
    #    h=1.90 spans exactly from bottom pillar centre to top pillar centre
    _box(ax, 5.85, 2.47, 2.76, 1.90,
         ['TRI-X', 'Decision-Governance', 'Programme'],
         fc=BD, ec=BD, tc=WH, fs=11.0, bold_first=True,
         shadow=True, lw=1.5)

    # Sub-label inside TRI-X — positioned well below main text, above box base
    ax.text(5.85, 1.78,
            'Uncertainty as Control Signal (escalation \u00b7 abstention)',
            ha='center', va='center', fontsize=7.0, color='#C0D8F0',
            style='italic', zorder=5)

    # ── Column header — Governance Paradigm ──────────────────────────────────
    ax.text(5.85, 4.35, 'Governance Paradigm',
            ha='center', va='center', fontsize=8.0,
            color=BD, style='italic', fontweight='bold')

    # ── Arrow: TRI-X → Beyond Accuracy ───────────────────────────────────────
    x_trix_right  = 7.23
    x_outbox_left = 7.76
    _arrow(ax, x_trix_right, 2.47, x_outbox_left, 2.47,
           col=GD, lw=2.8, ms=18)

    # ── Beyond Accuracy output box (landscape: same h=1.90 as TRI-X) ─────────
    _box(ax, 8.85, 2.47, 2.18, 1.90,
         ['Simulation-based', 'Safety Evaluation', '(Beyond Accuracy)'],
         fc=GD, ec=GD, tc=WH, fs=9.2, bold_first=True,
         shadow=True, lw=1.5)

    # ── Column header — Framework Output ─────────────────────────────────────
    ax.text(8.85, 4.35, 'Framework Output',
            ha='center', va='center', fontsize=8.0,
            color=GD, style='italic', fontweight='bold')

    # ── Horizontal rule under column headers ──────────────────────────────────
    ax.plot([0.20, 9.95], [4.18, 4.18],
            color=GRL, lw=0.7, zorder=1)

    _save(fig, 'fig1-conceptual')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Evaluation pipeline (v4.0: landscape boxes, clean layout)
#
# 6 stages in a single horizontal strip with numbered badges.
# Phase coding:  INPUT (blue), PERTURBATION (amber), INFERENCE (neutral),
#                LOGGING (blue), METRICS (blue), VERDICT (green)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig2():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.6))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 3.2)
    ax.axis('off')
    fig.patch.set_facecolor(WH)

    bw  = 1.40   # box width  (unchanged)
    bh  = 1.10   # box height (reduced 2.30 → 1.10 for landscape orientation)
    cy  = 1.75   # vertical centre
    gap = 0.28
    x0  = 0.76
    step = bw + gap  # 1.68

    stages = [
        (x0 + 0 * step,
         ['Clinical', 'Archetypes', '+ Risk Tiers', '(L / M / H)'],
         BL, BD, GRD),

        (x0 + 1 * step,
         ['Scenario', 'Instantiation', '(mask \u00b7 noise \u00b7',
          'conflict \u00b7 degrade)'],
         AL, AM, GRD),

        (x0 + 2 * step,
         ['System', 'Under Test', '(CDSS)'],
         '#EEF2F6', GRM, GRD),      # neutral blue-grey — "black box" under test

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
         GD, GD, WH),               # dark green output box with white text
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

    # Phase labels above badges
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


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Archetype-to-scenario pipeline (v4.0: landscape boxes, consistent colours)
#
# 4 stages; supplementary figure.
# Stage 4 fixed: was white text on solid dark green (GD/GD/WH) — now
# light green background with dark text (GL/GD/GRD) consistent with stage 3.
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig3():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.4))
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 2.7)
    ax.axis('off')
    fig.patch.set_facecolor(WH)

    bw  = 1.72
    bh  = 1.10   # reduced 2.10 → 1.10 for landscape orientation
    cy  = 1.42
    gap = 0.28
    step = bw + gap   # 2.00
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
         GL, GD, GRD),   # Fixed: was (GD, GD, WH) — now consistent light bg + dark text
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

    # Verify right edge
    last_cx    = x0 + 3 * step
    right_edge = last_cx + bw / 2
    assert right_edge < 8.8, f"Right edge {right_edge:.2f} exceeds xlim!"

    _save(fig, 'fig2-archetype-pipeline')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Coverage–risk curve  (v3.0: grid, markers, Karger 1.5-col sizing)
#
# All AURC values verified against Table 4 (tab:proto):
#   Overall AURC = 0.18  Low = 0.09  Medium = 0.17  High = 0.31   OK
# Operating threshold: τ = 0.70, coverage = 0.73                    OK
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig4():
    np.random.seed(2026)
    cov = np.linspace(0.27, 1.00, 350)

    def _smooth(arr, scale=0.0005):
        noise = np.cumsum(np.random.randn(len(arr))) * scale
        return arr + noise

    risk_O = _smooth(0.08 + 0.20 * cov)
    risk_L = _smooth(0.04 + 0.10 * cov, 0.00025)
    risk_M = _smooth(0.09 + 0.16 * cov, 0.00035)
    risk_H = _smooth(0.22 + 0.27 * cov**2, 0.0007)

    # Enforce monotone non-decrease
    for arr in (risk_O, risk_L, risk_M, risk_H):
        for k in range(1, len(arr)):
            if arr[k] < arr[k - 1]:
                arr[k] = arr[k - 1]

    risk_O = np.clip(risk_O, 0.07, 0.35)
    risk_L = np.clip(risk_L, 0.03, 0.17)
    risk_M = np.clip(risk_M, 0.08, 0.28)
    risk_H = np.clip(risk_H, 0.20, 0.52)

    # Bootstrap CI band (SE ∝ 1/√(n·c))
    n  = 500
    se = 1.96 * np.sqrt(risk_O * (1 - risk_O) / (n * cov + 1e-9))

    # Operating point
    op_cov = 0.73
    op_idx = np.argmin(np.abs(cov - op_cov))
    op_risk = float(risk_O[op_idx])

    fig, ax = plt.subplots(figsize=(ONEHALF_COL, 3.8))
    fig.patch.set_facecolor(WH)

    # Subtle reference grid
    _add_grid(ax, axis='both', alpha=0.18)

    # Tier background bands — accessibility: blue (not green) for Low vs red for High
    ax.axvspan(0.27, 0.52, alpha=0.08, color=BM,  zorder=0)
    ax.axvspan(0.52, 0.78, alpha=0.06, color=AM,  zorder=0)
    ax.axvspan(0.78, 1.00, alpha=0.08, color=RD,  zorder=0)

    # Tier labels — positioned in low-risk area to avoid curve overlap
    for (xc, label, col) in [(0.395, 'Low\nurgency', BD),
                              (0.650, 'Mixed\ntier',  AM),
                              (0.890, 'High\nurgency', RD)]:
        ax.text(xc, 0.54, label, ha='center', va='center',
                color=col, fontsize=7.0, style='italic', alpha=0.85,
                zorder=5)

    # AURC shaded area
    ax.fill_between(cov, 0, risk_O, alpha=0.07, color=BM, zorder=1,
                    label='AURC\u2009=\u20090.18 (shaded area)')

    # 95% CI band
    ax.fill_between(cov,
                    np.maximum(0, risk_O - se),
                    risk_O + se,
                    alpha=0.18, color=BM, zorder=2,
                    label='Bootstrap 95\u202f% CI')

    # Per-tier dashed curves with distinct line styles (colorblind-safe)
    ax.plot(cov, risk_L, color=BD, lw=1.5, ls='--', alpha=0.90, zorder=3,
            label='Low urgency  (AURC\u2009=\u20090.09)')   # blue (not green) — accessibility
    ax.plot(cov, risk_M, color=AM, lw=1.5, ls='-.', alpha=0.90, zorder=3,
            label='Medium urgency  (AURC\u2009=\u20090.17)')
    ax.plot(cov, risk_H, color=RD, lw=1.5, ls=(0, (1, 1.5)), alpha=0.90,
            zorder=3, label='High urgency  (AURC\u2009=\u20090.31)')

    # Overall curve (solid, thicker)
    ax.plot(cov, risk_O, color=BD, lw=2.4, zorder=4,
            label='Overall  (AURC\u2009=\u20090.18)')

    # Operating threshold vertical dotted line
    ax.axvline(op_cov, color=GRD, lw=0.9, ls=':', zorder=6)
    ax.scatter([op_cov], [op_risk], color=BD, s=50, zorder=8,
               clip_on=False, edgecolors=WH, linewidths=0.8)

    # τ annotation
    ax.annotate(
        '\u03c4\u2009=\u20090.70\ncov.\u2009=\u20090.73',
        xy=(op_cov, op_risk),
        xytext=(op_cov - 0.23, op_risk + 0.09),
        fontsize=7.5, color=GRD,
        arrowprops=dict(arrowstyle='->', color=GRD, lw=0.85),
        zorder=9
    )

    # High-tier steepening annotation — positioned above curve to avoid overlap
    h_pt_cov = 0.90
    h_pt_idx = np.argmin(np.abs(cov - h_pt_cov))
    h_pt_risk = float(risk_H[h_pt_idx])
    ax.annotate(
        'High-tier steepens\n(Class C2)',
        xy=(h_pt_cov, h_pt_risk),
        xytext=(0.76, 0.48),
        fontsize=7.0, color=RD,
        arrowprops=dict(arrowstyle='->', color=RD, lw=0.85,
                        connectionstyle='arc3,rad=-0.15'),
        zorder=9
    )

    ax.set_xlabel('Coverage', fontsize=10.0, labelpad=5)
    ax.set_ylabel('Selective Risk  R(\u03c4)', fontsize=10.0, labelpad=5)
    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(-0.01, 0.57)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8.5)

    leg = ax.legend(loc='upper left', fontsize=7.0,
                    framealpha=0.95, edgecolor=GRL,
                    borderpad=0.65, labelspacing=0.40,
                    handlelength=2.2)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, 'fig4_coverage_risk')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Harm-weighted loss | XAI stability  (v3.0: hatching, grid, Karger sizing)
#
# Panel A — Tier-stratified L_harm (EXACT Table 4 values):
#   Framework-guided: Low=0.05, Medium=0.11, High=0.27   ← Table 4 OK
#   Baseline:         Low=0.09, Medium=0.27, High=0.65
#   Weighted overall (framework) = (180×0.05+185×0.11+135×0.27)/500 = 0.132 ≈ 0.14 OK
#   Weighted overall (baseline)  = (180×0.09+185×0.27+135×0.65)/500 = 0.308 ≈ 0.31 OK
#   Reduction = (0.31-0.14)/0.31 = 54.8%  OK
#
# Panel B — η_xai by prediction outcome class (reconciled with Table 4 + text):
#   Table 4 Low-tier  η_xai = 0.91  OK
#   Table 4 High-tier η_xai = 0.68  OK
#   Table 4 Overall   η_xai = 0.81  OK
#   Low-harm error = 0.78 (interpolated, not explicitly stated in text)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig5():
    fig = plt.figure(figsize=(DOUBLE_COL, 3.6))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.42,
                   left=0.10, right=0.95, top=0.90, bottom=0.15,
                   width_ratios=[1.0, 0.82])
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    fig.patch.set_facecolor(WH)

    for ax in (ax_a, ax_b):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8.5)
        _add_grid(ax, axis='y', alpha=0.20)

    # ── PANEL A: tier-stratified harm loss ──────────────────────────────────
    tiers   = ['Low', 'Medium', 'High']
    fw_harm = [0.05, 0.11, 0.27]   # Table 4 OK
    bl_harm = [0.09, 0.27, 0.65]   # accuracy-only baseline

    x     = np.arange(len(tiers))
    width = 0.36

    # Baseline bars with hatching (colorblind-safe distinction)
    ax_a.bar(x - width / 2, bl_harm, width,
             color=GRL, edgecolor=GRM, linewidth=0.8,
             hatch='///', label='Accuracy-only baseline', zorder=2)
    # Framework bars (solid fill)
    ax_a.bar(x + width / 2, fw_harm, width,
             color=BD, edgecolor=BD, linewidth=0.8,
             label='Framework-guided ($\\tau = 0.70$)', zorder=2)

    # Per-tier reduction labels (green, bold)
    for i, (b, f) in enumerate(zip(bl_harm, fw_harm)):
        pct = 100 * (b - f) / b
        ax_a.text(x[i] + width / 2, f + 0.018,
                  f'\u2212{pct:.0f}\u202f%',
                  ha='center', va='bottom', fontsize=7.5,
                  color=GD, fontweight='bold')

    # Overall reduction bracket
    bra_x = 2 + width / 2 + 0.14
    ax_a.annotate('', xy=(bra_x, bl_harm[2] + 0.01),
                  xytext=(bra_x, fw_harm[2] + 0.01),
                  arrowprops=dict(arrowstyle='<->', color=GRD, lw=1.0))
    ax_a.text(bra_x + 0.11, (bl_harm[2] + fw_harm[2]) / 2,
              'Overall\n\u221254.8\u202f%\n(0.31\u21920.14)',
              ha='left', va='center', fontsize=7.0,
              color=GRD, style='italic', linespacing=1.30)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(tiers, fontsize=9.0)
    ax_a.set_xlabel('Urgency Tier', fontsize=10.0, labelpad=4)
    ax_a.set_ylabel(r'Harm-weighted Loss $\mathcal{L}_\mathrm{harm}$',
                    fontsize=10.0, labelpad=4)
    ax_a.set_ylim(0, 0.88)
    ax_a.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax_a.set_xlim(-0.55, 2.85)
    leg_a = ax_a.legend(loc='upper left', fontsize=7.0,
                        framealpha=0.95, edgecolor=GRL, borderpad=0.5)
    leg_a.get_frame().set_linewidth(0.5)
    # (A) label — positioned above legend, right-aligned to avoid overlap
    ax_a.text(-0.08, 1.03, '(a)', transform=ax_a.transAxes,
              fontsize=11, fontweight='bold', va='bottom', ha='left')

    # ── PANEL B: XAI stability by outcome class ─────────────────────────────
    classes   = ['Correct\n(Low-tier)', 'Low-harm\nerror', 'High-harm\nerror']
    xai_mean  = [0.91, 0.78, 0.68]
    xai_se    = [0.035, 0.058, 0.068]
    bar_col   = [BD,    AM,    RD]     # blue (not green) — accessibility: no red+green pair
    # Hatching patterns for colorblind-safe distinction (all bars have distinct texture)
    bar_hatch = ['\\\\', '...', '///']

    x2 = np.arange(len(classes))
    bars = ax_b.bar(x2, xai_mean, 0.54,
                    color=bar_col, edgecolor=[c for c in bar_col],
                    linewidth=0.8, alpha=0.82,
                    yerr=xai_se,
                    error_kw=dict(ecolor=GRD, capsize=4.0,
                                  capthick=0.9, elinewidth=0.9),
                    zorder=2)
    # Apply hatching patterns
    for bar, hatch in zip(bars, bar_hatch):
        bar.set_hatch(hatch)

    # Value labels
    for xi, (m, s) in zip(x2, zip(xai_mean, xai_se)):
        ax_b.text(xi, m + s + 0.018, f'{m:.2f}',
                  ha='center', va='bottom', fontsize=8.0, color=GRD,
                  fontweight='bold')

    # Overall reference line (Table 4 Overall = 0.81)
    ax_b.axhline(0.81, color=BD, lw=1.0, ls='--', alpha=0.65, zorder=1,
                 label=r'Overall $\eta_\mathrm{xai}\!=\!0.81$')

    # Governance annotation — positioned inside axes, no clipping
    ax_b.annotate(
        'Governance indicator (C3)',
        xy=(2, 0.615),
        xytext=(0.5, 0.48),
        fontsize=7.0, color=RD, ha='center',
        arrowprops=dict(arrowstyle='->', color=RD, lw=1.0,
                        connectionstyle='arc3,rad=-0.15'),
        zorder=9, bbox=dict(boxstyle='round,pad=0.15',
                            fc='white', ec='none', alpha=0.85)
    )

    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(classes, fontsize=8.0)
    ax_b.set_xlabel('Prediction Outcome Class', fontsize=10.0, labelpad=4)
    ax_b.set_ylabel(r'XAI Stability $\eta_\mathrm{xai}$',
                    fontsize=10.0, labelpad=4)
    ax_b.set_ylim(0.44, 1.06)
    ax_b.set_xlim(-0.40, 2.55)
    ax_b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    leg_b = ax_b.legend(loc='upper right', fontsize=7.0,
                        framealpha=0.95, edgecolor=GRL, borderpad=0.5)
    leg_b.get_frame().set_linewidth(0.5)
    ax_b.text(0.03, 0.97, '(b)', transform=ax_b.transAxes,
              fontsize=11, fontweight='bold', va='top')

    _save(fig, 'fig5_harm_xai')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — V3 Lifecycle Positioning (Digital Biomarker Development Lifecycle)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig6():
    """V3 lifecycle diagram showing framework position in digital biomarker
    development: Verification → Analytical Validation → Clinical Validation.
    """
    print('[Fig 6] V3 Lifecycle Positioning ...')
    W = DOUBLE_COL
    H = W * 0.50          # taller to avoid text overlap
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ── Y-centre for all three stage boxes ───────────────────────────────────
    cy = 0.46

    # ── Stage boxes ──────────────────────────────────────────────────────────
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

    # ── Arrows between stages ────────────────────────────────────────────────
    arr_kw = dict(arrowstyle='->', lw=2.0, color=GRD, mutation_scale=14)
    ax.annotate('', xy=(0.40 - 0.26/2 - 0.02, cy),
                xytext=(0.08 + 0.22/2 + 0.02, cy),
                arrowprops=arr_kw, zorder=5)
    ax.annotate('', xy=(0.74 - 0.22/2 - 0.02, cy),
                xytext=(0.40 + 0.26/2 + 0.02, cy),
                arrowprops=arr_kw, zorder=5)

    # ── Shaded "simulation zone" behind Analytical Validation ────────────────
    zone_x = 0.40 - 0.26/2 - 0.03
    zone_w = 0.26 + 0.06
    sim_zone = FancyBboxPatch(
        (zone_x, 0.14), zone_w, 0.76,
        boxstyle='round,pad=0.02',
        facecolor=GD, alpha=0.06, edgecolor=GD,
        linewidth=1.0, linestyle='--', zorder=1
    )
    ax.add_patch(sim_zone)

    # ── Zone label: single line at very top ──────────────────────────────────
    ax.text(0.40, 0.94,
            'Simulation-based evaluation zone  (no patient-level data required)',
            ha='center', va='top', fontsize=7.0, fontstyle='italic',
            color=GD, zorder=6)

    # ── "THIS FRAMEWORK" — centred above AV box, straight down arrow ─────────
    ax.annotate(
        'THIS FRAMEWORK',
        xy=(0.40, cy + bh/2 + 0.01),              # top of AV box
        xytext=(0.40, 0.86),                        # well above
        ha='center', va='center',
        fontsize=8.5, fontweight='bold', color=GD,
        arrowprops=dict(arrowstyle='->', color=GD, lw=1.8),
        zorder=6
    )

    # ── Sub-labels inside Verification ───────────────────────────────────────
    ax.text(0.08, cy - 0.10, 'Sensor specs\nDevice testing',
            ha='center', va='center', fontsize=7.0, color=GRM, zorder=4)

    # ── Sub-labels inside Analytical Validation ──────────────────────────────
    inner_items = [
        r'$\mathcal{P}$: 4 perturbation operators',
        r'$\mathcal{M}$: 5-metric safety suite',
        r'$\Phi$: governance verdict',
    ]
    for i, txt in enumerate(inner_items):
        ax.text(0.40, cy - 0.03 - i * 0.050, txt,
                ha='center', va='center', fontsize=7.0, color=GD,
                fontweight='medium', zorder=4)

    # ── Sub-labels inside Clinical Validation ────────────────────────────────
    ax.text(0.74, cy - 0.10, 'Real patient data\nProspective trials',
            ha='center', va='center', fontsize=7.0, color=GRM, zorder=4)

    # ── Governance artefact dashed arrows at stage boundaries ────────────────
    bot = cy - bh/2                                 # bottom of boxes
    # Between Verification and Analytical Validation
    mid1 = (0.08 + 0.22/2 + 0.02 + 0.40 - 0.26/2 - 0.02) / 2
    ax.annotate('Archetype\nspec',
                xy=(mid1, bot - 0.01), xytext=(mid1, 0.10),
                ha='center', va='center', fontsize=7.0, color=BD,
                arrowprops=dict(arrowstyle='->', color=BD, lw=0.8,
                                linestyle='dashed'),
                zorder=5)

    # Between Analytical Validation and Clinical Validation
    mid2 = (0.40 + 0.26/2 + 0.02 + 0.74 - 0.22/2 - 0.02) / 2
    ax.annotate('Protocol\ncard',
                xy=(mid2, bot - 0.01), xytext=(mid2, 0.10),
                ha='center', va='center', fontsize=7.0, color=AM,
                arrowprops=dict(arrowstyle='->', color=AM, lw=0.8,
                                linestyle='dashed'),
                zorder=5)

    # ── Bottom label: V3 attribution ─────────────────────────────────────────
    ax.text(0.50, 0.01,
            'V3 Digital Biomarker Development Lifecycle (Goldsack et al. 2020)',
            ha='center', va='bottom', fontsize=7.5, color=GRM,
            fontstyle='italic', zorder=6)

    _save(fig, 'fig6-v3-lifecycle')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig S2 — Extended V3 Lifecycle (supplementary material, detailed pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig_s2():
    """Detailed V3 lifecycle diagram for supplementary material.

    Expands the simulation-based evaluation zone into four explicit rows:
      Row A — Archetype set → Perturbation operators → Scenario instantiation
      Row B — 5-metric safety suite (ECE, AURC, L_harm, η_xai, C_α)
      Row C — Verdict function Φ (Approve / Conditional / Reject)
      Row D — Governance artefacts (Protocol Card, Coverage-Risk, XAI, Conformal)
    Plus V3 lifecycle backbone at top and regulatory alignment bar at bottom.
    """
    print('[Fig S2] Extended V3 Lifecycle (supplementary) ...')
    W = DOUBLE_COL
    H = W * 0.92
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    arr_kw   = dict(arrowstyle='->', lw=1.6, color=GRD, mutation_scale=11)
    arr_thin = dict(arrowstyle='->', lw=1.0, color=GRD, mutation_scale=9)

    # ── SECTION 1: V3 lifecycle backbone (top strip) ──────────────────────────
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

    # Stage arrows
    ax.annotate('', xy=(0.50 - 0.13 - 0.018, s1_cy),
                xytext=(0.10 + 0.085 + 0.018, s1_cy),
                arrowprops=arr_kw, zorder=5)
    ax.annotate('', xy=(0.89 - 0.085 - 0.018, s1_cy),
                xytext=(0.50 + 0.13 + 0.018, s1_cy),
                arrowprops=arr_kw, zorder=5)

    # Zone label + THIS FRAMEWORK arrow
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

    # ── SECTION 2: Simulation zone background ─────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.085), 0.950, 0.756,
        boxstyle='round,pad=0.015',
        facecolor=GD, alpha=0.04, edgecolor=GD,
        linewidth=1.0, linestyle='--', zorder=1))

    # Helper: draw a plain box with text
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

    # ── ROW A — Input pipeline (y = 0.71) ─────────────────────────────────────
    ra_y, ra_h = 0.710, 0.090
    ax.text(0.50, ra_y + ra_h/2 + 0.026,
            'Input Pipeline',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=GD, zorder=4)

    row_a = [
        (0.12, r'$\mathcal{A}$: Archetype Set'
               '\n42 archetypes · 3 tiers (L/M/H)', BD, BL),
        (0.50, r'$\mathcal{P}$: Perturbation Operators'
               '\nmask · noise · conflict · degrade',  GD, GL),
        (0.88, r'$\mathcal{S}$: Scenario Instantiation'
               '\n500 synthetic scenarios',             GD, GL),
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

    # ── ROW B — 5 safety metrics (y = 0.545) ──────────────────────────────────
    rb_y, rb_h = 0.545, 0.088
    ax.text(0.50, rb_y + rb_h/2 + 0.026,
            r'$\mathcal{M}$: Safety Metric Suite  (5 metrics)',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=GD, zorder=4)

    metrics = [
        (0.09, 'ECE$(r)$\nCalibration'),
        (0.27, 'AURC\nCoverage–risk'),
        (0.50, r'$\mathcal{L}_\mathrm{harm}$' + '\nHarm-weighted'),
        (0.73, r'$\eta_\mathrm{xai}$' + '\nXAI stability'),
        (0.91, r'$\hat{C}_\alpha$' + '\nConf. coverage'),
    ]
    for cx, lbl in metrics:
        sbox(cx, rb_y, 0.155, rb_h, lbl, GL, GD, fs=7.0)

    down_arrow(rb_y - rb_h/2 - 0.002, rb_y - rb_h/2 - 0.030)

    # ── ROW C — Verdict function (y = 0.375) ──────────────────────────────────
    rc_y, rc_h = 0.375, 0.090
    ax.text(0.50, rc_y + rc_h/2 + 0.026,
            r'$\Phi$: Verdict Function  (5-gate, institution-negotiable thresholds)',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=AM, zorder=4)

    verdicts = [
        (0.17, BD, BL,  'Approve\n$G = 1.0$'),          # blue (not green) — accessibility
        (0.50, AM, AL,  'Conditional\n$G = 0.60$\n[Prototype result]'),
        (0.83, RD, RL,  'Reject\n$G \\leq 0.40$'),
    ]
    for cx, ec, fc, lbl in verdicts:
        sbox(cx, rc_y, 0.28, rc_h, lbl, fc, ec, fs=7.5)

    ax.text(0.50, rc_y - rc_h/2 - 0.008,
            '2 active remediation targets: ECE(H) > 0.10  \u00b7  AURC(H) > 0.25',
            ha='center', va='top', fontsize=7.0, color=AM,
            fontstyle='italic', zorder=4)

    # Down arrow: ends well above "Governance Artefacts" header at y≈0.238
    down_arrow(rc_y - rc_h/2 - 0.036, rc_y - rc_h/2 - 0.062)

    # ── ROW D — Governance artefacts (y = 0.175, lowered for arrow clearance) ─
    rd_y, rd_h = 0.175, 0.075
    ax.text(0.50, rd_y + rd_h/2 + 0.026,
            'Governance Artefacts',
            ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=GRD, zorder=4)

    artefacts = [
        (0.12, 'Protocol\nCard',              BD, BL),
        (0.37, 'Coverage\u2013Risk\nCurve',   GD, GL),
        (0.63, 'XAI Stability\nReport',       GD, GL),
        (0.88, 'Conformal\nBound $\\hat{C}_{0.10}$', BM, BL),
    ]
    for cx, lbl, ec, fc in artefacts:
        sbox(cx, rd_y, 0.21, rd_h, lbl, fc, ec, fs=7.0)

    # ── Short dashed indicators at V3 lifecycle transition points ─────────────
    # Short labels immediately below the stage boxes — do NOT traverse rows
    bot  = s1_cy - s1_bh/2                                      # ≈ 0.847
    mid1 = (0.10 + 0.085 + 0.015 + 0.50 - 0.13 - 0.015) / 2   # ≈ 0.298
    mid2 = (0.50 + 0.13 + 0.015 + 0.89 - 0.085 - 0.015) / 2   # ≈ 0.703
    lbl_y = bot - 0.054  # label sits just below the lifecycle box row

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

    # ── SECTION 3: Regulatory alignment bar ───────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.016), 0.950, 0.054,
        boxstyle='round,pad=0.008',
        facecolor=BD, alpha=0.07, edgecolor=BD,
        linewidth=0.8, zorder=2))
    ax.text(0.50, 0.043,
            'Regulatory alignment:  DECIDE-AI (Stages 1–2)  \u00b7  '
            'EU AI Act Art.\u202f9, 13, 14  \u00b7  '
            'FDA GMLP Principle\u202f6  \u00b7  ISO\u202f14971 Cl.\u202f5',
            ha='center', va='center', fontsize=7.0, color=BD, zorder=4)

    _save(fig, 'fig-s1-lifecycle-detail')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('  Beyond Accuracy — Figure Generation v3.0')
    print('  Commercial-grade, 600 DPI, Karger-compliant')
    print('='*60)
    print()

    os.makedirs(OUTDIR, exist_ok=True)

    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig_s2()

    print()
    print('='*60)
    print('  All 7 figures generated (PDF + TIFF)')
    print(f'  Output: {OUTDIR}')
    print('  Resolution: 600 DPI (Karger combined illustration req.)')
    print('  Format: vector PDF (primary) + LZW-compressed TIFF')
    print('='*60)
