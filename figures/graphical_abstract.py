"""
Graphical Abstract - Beyond Accuracy
Renders the summary graphic into the repository's figures/ directory.

The "Key Findings" column and the verdict band are read at run time from
results/eval_results.json (written by run_all.py under seed 42); no metric
value is hardcoded. Run `python run_all.py` first, then this script.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Shared, byte-identical publication style/util (vendored from _management).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubviz import apply_pub_style, PALETTE, load_results  # noqa: E402

# -- Paths: write into the repo figures/ dir; read from results/ -------------
REPO = os.path.dirname(HERE)
R = load_results('eval_results.json')

# -- Pull real values --------------------------------------------------------
ece_high    = R['calibration']['ece_by_tier']['High']
harm_fw     = R['harm']['harm_weighted_loss_framework']
harm_bl     = R['harm']['harm_weighted_loss_baseline']
harm_red    = 100.0 * R['harm']['reduction_fraction']
eta_xai     = R['explanation_stability']['eta_xai_spearman']
conf_cov    = R['conformal']['empirical_coverage']
conf_alpha  = R['conformal']['alpha']
gov_score   = R['governance']['score']
gov_verdict = R['governance']['verdict']
n_failed    = len(R['governance'].get('failed_gates', []))

# -- Karger specifications ---------------------------------------------------
DPI = 300
W_CM, H_CM = 32, 14
W_IN, H_IN = W_CM / 2.54, H_CM / 2.54

# Canonical Top-Tier style (shared, vendored in pubviz). This is a single
# full-canvas schematic, so constrained_layout is disabled to preserve the
# explicit coordinate layout; branded box fills are kept (FIGURE_STYLE rule 7)
# over the shared fonts.
import matplotlib as mpl  # noqa: E402

apply_pub_style()
mpl.rcParams['figure.constrained_layout.use'] = False

# -- Colour palette (branded schematic fills; text/accents map to Okabe-Ito) --
BLU   = '#1C4E80'
GRN   = '#007040'
RED   = '#B41414'
AMB   = '#C06800'
TEAL  = '#005F73'
LBLU  = '#D6E4F5'
LGRN  = '#D1F0E0'
LRED  = '#FAD9D9'
LAMB  = '#FFF0D0'
LTEAL = '#D0F0F5'
BG    = '#FAFCFF'
DGRY  = '#2D2D2D'
MGRY  = '#555555'
LGRY  = '#E8E8E8'

fig, ax = plt.subplots(1, 1, figsize=(W_IN, H_IN), facecolor=BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis('off')
fig.patch.set_facecolor(BG)


# -- Helpers -----------------------------------------------------------------
def rbox(cx, cy, w, h, fc, ec, text, fs=9.0, fw='normal', tc=None, lw=1.2):
    tc = tc if tc else ec
    x0, y0 = cx - w/2, cy - h/2
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.30',
        facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, linespacing=1.30)


def varrow(x, y_from, y_to, col=MGRY, lw=1.1):
    ax.annotate('', xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                connectionstyle='arc3,rad=0'))


def harrow(x_from, x_to, y, col=BLU, lw=1.5):
    ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                connectionstyle='arc3,rad=0'))


# -- Column layout -----------------------------------------------------------
C1L, C1R, C1W, C1X = 2,  30, 28, 16
C2L, C2R, C2W, C2X = 33, 61, 28, 47
C3L, C3R, C3W, C3X = 64, 98, 34, 81

# -- Row layout (5 content rows, aligned across all 3 columns) ---------------
BH    = 4.80
GAP   = 1.20
STEP  = BH + GAP
TOP   = 35.5
ROW   = [TOP - i * STEP for i in range(5)]

VH    = 3.20
VY    = ROW[4] - BH/2 - GAP/2 - VH/2

# -- Title -------------------------------------------------------------------
ax.text(50, 42.6,
        'Beyond Accuracy: Simulation-Based Safety Evaluation'
        ' for Clinical AI and Digital Biomarker Systems',
        ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=BLU)
ax.plot([2, 98], [40.9, 40.9], color=BLU, lw=0.6, alpha=0.35)

# -- Column headers ----------------------------------------------------------
ax.text(C1X, 39.4,
        r'Framework  $(\mathcal{A},\mathcal{P},\mathcal{S},\mathcal{M},\Phi)$',
        ha='center', va='center', fontsize=10, fontweight='bold', color=BLU)
ax.text(C2X, 39.4, 'Evaluation Pipeline',
        ha='center', va='center', fontsize=10, fontweight='bold', color=BLU)
ax.text(C3X, 39.4, 'Key Findings & Lifecycle Position',
        ha='center', va='center', fontsize=10, fontweight='bold', color=BLU)

# -- Column separators -------------------------------------------------------
for sx in [31.5, 62.5]:
    ax.plot([sx, sx], [5.8, 39.0], color=BLU, lw=0.5, alpha=0.18, ls='--')

# ============================================================================
# COLUMN 1 - Framework definition (schematic; no metric values)
# ============================================================================
c1 = [
    (ROW[0], LBLU, BLU,
     'Archetypes (A)\nclinical scenarios + risk tiers'),
    (ROW[1], LBLU, BLU,
     'Perturbation Operators (P)\nmask · noise · conflict · degrade'),
    (ROW[2], LBLU, BLU,
     'Scenarios (S)\ncontrolled uncertainty profiles'),
    (ROW[3], LGRN, GRN,
     'Safety Metrics (M)\nECE, AURC, harm loss, XAI, conformal'),
    (ROW[4], LAMB, AMB,
     'Verdict Function (Phi)\n5-gate governance decision'),
]
for cy, fc, ec, txt in c1:
    rbox(C1X, cy, C1W, BH, fc, ec, txt, fs=8.5, fw='bold', tc=ec)

for i in range(len(c1) - 1):
    varrow(C1X, ROW[i] - BH/2 - 0.08, ROW[i+1] + BH/2 + 0.08, col='#AAAAAA', lw=0.9)

# ============================================================================
# COLUMN 2 - Evaluation pipeline (schematic; cohort size from results)
# ============================================================================
n_scen = R['meta']['cohort_n']
c2 = [
    (ROW[0], LBLU, BLU,   'Expert-validated\narchetypes'),
    (ROW[1], LBLU, BLU,   'Controlled\nperturbation (4 operators)'),
    (ROW[2], LBLU, BLU,   f'{n_scen} synthetic\nscenarios'),
    (ROW[3], LGRY, DGRY,  'CDSS under test\n(inference + logging)'),
    (ROW[4], LGRN, GRN,   '5 safety metrics\n+ governance verdict'),
]
for cy, fc, ec, txt in c2:
    rbox(C2X, cy, C2W, BH, fc, ec, txt, fs=9.0)

for i in range(len(c2) - 1):
    varrow(C2X, ROW[i] - BH/2 - 0.08, ROW[i+1] + BH/2 + 0.08, col=BLU, lw=1.2)

# ============================================================================
# COLUMN 3 - Key findings (values read from results/)
# ============================================================================
c3 = [
    (ROW[0], LRED,  RED,
     f'ECE(H) = {ece_high:.3f}\nLargest calibration error in high-urgency tier'),
    (ROW[1], LGRN,  GRN,
     f'Harm loss: {harm_fw:.3f} vs {harm_bl:.3f} baseline\n{harm_red:.1f}% reduction'),
    (ROW[2], LAMB,  AMB,
     f'Explanation stability eta_xai = {eta_xai:.3f}\n(rank-correlation under perturbation)'),
    (ROW[3], LTEAL, TEAL,
     f'Conformal coverage {conf_cov:.3f} at alpha={conf_alpha:.2f}\nfinite-sample bound'),
    (ROW[4], LGRN,  GRN,
     'V3 Lifecycle: Analytical Validation stage\nno patient-level data required'),
]
for cy, fc, ec, txt in c3:
    rbox(C3X, cy, C3W, BH, fc, ec, txt, fs=8.5, tc=ec)

# Verdict box (values from results/)
rbox(C3X, VY, C3W, VH, '#FFF3CD', AMB,
     f'Verdict: {gov_verdict.upper()}  |  G = {gov_score:.2f}  |  '
     f'{n_failed} remediation targets',
     fs=9.5, fw='bold', tc=AMB, lw=1.8)

# ============================================================================
# ARROWS: col 2 -> col 3 (one per row, aligned)
# ============================================================================
for cy in ROW:
    harrow(C2R + 0.4, C3L - 0.4, cy)

ax.annotate('',
            xy=(C3L - 0.4, VY),
            xytext=(C2R + 0.4, ROW[4] - BH/2 - 0.1),
            arrowprops=dict(arrowstyle='->', color=AMB, lw=1.4,
                            connectionstyle='angle,angleA=0,angleB=90,rad=2'))

# ============================================================================
# BOTTOM BAR - governance artefacts + regulatory alignment
# ============================================================================
ax.plot([2, 98], [5.0, 5.0], color=BLU, lw=0.8, alpha=0.45)
ax.text(50, 2.6,
        'Governance artefacts: Protocol Card  |  Coverage-Risk Curves'
        '  |  XAI Stability Reports'
        '  ·  Regulatory: DECIDE-AI  ·  EU AI Act'
        '  ·  FDA GMLP  ·  ISO 14971',
        ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color=MGRY)

# -- Save --------------------------------------------------------------------
# Canonical: vector PDF + 300-dpi PNG (same basename). JPG kept for the Karger
# graphical-abstract portal upload requirement.
base = os.path.join(HERE, 'graphical-abstract')
fig.savefig(base + '.pdf', format='pdf', facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)
fig.savefig(base + '.png', format='png', dpi=DPI, facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)
outpath = base + '.jpg'
fig.savefig(outpath, dpi=DPI, format='jpg', facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)
plt.close()

from PIL import Image
im = Image.open(outpath)
print(f'Saved: {base}.pdf (vector), {base}.png (300 dpi), {outpath}')
print(f'Size: {im.size[0]} x {im.size[1]} px  (max 4800 x 2100)')
print(f'DPI: {im.info.get("dpi", "not set")}')
print('OK' if im.size[0] <= 4800 and im.size[1] <= 2100
      else 'WARNING: exceeds Karger max size')
