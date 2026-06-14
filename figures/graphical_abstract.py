"""
Graphical Abstract — Beyond Accuracy  (v4.0)
Karger Digital Biomarkers: JPG, max 4800x2100px, 300 dpi, Arial 9-12pt

v4.0: Complete redesign — perfect row alignment across all 3 columns,
      clean horizontal arrows col2→col3 at identical y-centres,
      consistent box heights, modern top-tier journal layout.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Karger specifications ────────────────────────────────────────────────────
DPI = 300
W_CM, H_CM = 32, 14
W_IN, H_IN = W_CM / 2.54, H_CM / 2.54   # ~12.6 × 5.51 inches

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Segoe UI', 'Helvetica', 'DejaVu Sans'],
    'font.size':       10,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
    'mathtext.fontset':'dejavusans',
})

# ── Colour palette ───────────────────────────────────────────────────────────
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


# ── Helpers ──────────────────────────────────────────────────────────────────
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


# ── Column layout ─────────────────────────────────────────────────────────────
C1L, C1R, C1W, C1X = 2,  30, 28, 16
C2L, C2R, C2W, C2X = 33, 61, 28, 47
C3L, C3R, C3W, C3X = 64, 98, 34, 81

# ── Row layout (5 content rows, perfectly aligned across all 3 columns) ──────
BH    = 4.80          # content box height
GAP   = 1.20          # gap between boxes
STEP  = BH + GAP      # 6.00 centre-to-centre
TOP   = 35.5          # top row centre
ROW   = [TOP - i * STEP for i in range(5)]   # [35.5, 29.5, 23.5, 17.5, 11.5]

VH    = 3.20          # verdict box height
VY    = ROW[4] - BH/2 - GAP/2 - VH/2        # verdict centre, below row 5

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(50, 42.6,
        'Beyond Accuracy: Simulation-Based Safety Evaluation'
        ' for Clinical AI and Digital Biomarker Systems',
        ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=BLU)
ax.plot([2, 98], [40.9, 40.9], color=BLU, lw=0.6, alpha=0.35)

# ── Column headers ─────────────────────────────────────────────────────────────
ax.text(C1X, 39.4,
        r'Framework  $(\mathcal{A},\mathcal{P},\mathcal{S},\mathcal{M},\Phi)$',
        ha='center', va='center', fontsize=10, fontweight='bold', color=BLU)
ax.text(C2X, 39.4, 'Evaluation Pipeline',
        ha='center', va='center', fontsize=10, fontweight='bold', color=BLU)
ax.text(C3X, 39.4, 'Key Findings & Lifecycle Position',
        ha='center', va='center', fontsize=10, fontweight='bold', color=BLU)

# ── Column separators ─────────────────────────────────────────────────────────
for sx in [31.5, 62.5]:
    ax.plot([sx, sx], [5.8, 39.0], color=BLU, lw=0.5, alpha=0.18, ls='--')

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 — Framework definition
# ═══════════════════════════════════════════════════════════════════════════════
c1 = [
    (ROW[0], LBLU, BLU,
     'Archetypes (A)\nclinical scenarios + risk tiers'),
    (ROW[1], LBLU, BLU,
     'Perturbation Operators (P)\nmask \u00b7 noise \u00b7 conflict \u00b7 degrade'),
    (ROW[2], LBLU, BLU,
     'Scenarios (S)\ncontrolled uncertainty profiles'),
    (ROW[3], LGRN, GRN,
     'Safety Metrics (M)\nECE, AURC, harm loss, XAI, conformal'),
    (ROW[4], LAMB, AMB,
     'Verdict Function (\u03a6)\n5-gate governance decision'),
]
for cy, fc, ec, txt in c1:
    rbox(C1X, cy, C1W, BH, fc, ec, txt, fs=8.5, fw='bold', tc=ec)

for i in range(len(c1) - 1):
    varrow(C1X, ROW[i] - BH/2 - 0.08, ROW[i+1] + BH/2 + 0.08, col='#AAAAAA', lw=0.9)

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 2 — Evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════════════
c2 = [
    (ROW[0], LBLU, BLU,   'Expert-validated\narchetypes'),
    (ROW[1], LBLU, BLU,   'Controlled\nperturbation (4 operators)'),
    (ROW[2], LBLU, BLU,   '500 synthetic\nscenarios'),
    (ROW[3], LGRY, DGRY,  'CDSS under test\n(inference + logging)'),
    (ROW[4], LGRN, GRN,   '5 safety metrics\n+ governance verdict'),
]
for cy, fc, ec, txt in c2:
    rbox(C2X, cy, C2W, BH, fc, ec, txt, fs=9.0)

for i in range(len(c2) - 1):
    varrow(C2X, ROW[i] - BH/2 - 0.08, ROW[i+1] + BH/2 + 0.08, col=BLU, lw=1.2)

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 — Key findings (aligned row-by-row with col 2)
# ═══════════════════════════════════════════════════════════════════════════════
c3 = [
    (ROW[0], LRED,  RED,   'ECE(H) = 0.131\nOverconfidence in high-urgency tier'),
    (ROW[1], LGRN,  GRN,   'Harm loss: 0.14 vs 0.31 baseline\n54.8% reduction'),
    (ROW[2], LAMB,  AMB,   '81% of high-harm misclassifications\nshow XAI instability'),
    (ROW[3], LTEAL, TEAL,  'Conformal coverage 91.3% at \u03b1=0.10\nfinite-sample bound'),
    (ROW[4], LGRN,  GRN,   'V3 Lifecycle: Analytical Validation stage\nno patient-level data required'),
]
for cy, fc, ec, txt in c3:
    rbox(C3X, cy, C3W, BH, fc, ec, txt, fs=8.5, tc=ec)

# Verdict box (full col 3 width, highlighted)
rbox(C3X, VY, C3W, VH, '#FFF3CD', AMB,
     'Verdict: CONDITIONAL  |  G\u2009=\u20090.60  |  2 remediation targets',
     fs=9.5, fw='bold', tc=AMB, lw=1.8)

# ═══════════════════════════════════════════════════════════════════════════════
# ARROWS: col 2 → col 3 (one per row, perfectly aligned)
# ═══════════════════════════════════════════════════════════════════════════════
for cy in ROW:
    harrow(C2R + 0.4, C3L - 0.4, cy)

# Arrow: col 2 last box → verdict (angled down)
ax.annotate('',
            xy=(C3L - 0.4, VY),
            xytext=(C2R + 0.4, ROW[4] - BH/2 - 0.1),
            arrowprops=dict(arrowstyle='->', color=AMB, lw=1.4,
                            connectionstyle='angle,angleA=0,angleB=90,rad=2'))

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM BAR — governance artefacts + regulatory alignment
# ═══════════════════════════════════════════════════════════════════════════════
ax.plot([2, 98], [5.0, 5.0], color=BLU, lw=0.8, alpha=0.45)
ax.text(50, 2.6,
        'Governance artefacts: Protocol Card  |  Coverage\u2013Risk Curves'
        '  |  XAI Stability Reports'
        '  \u00b7  Regulatory: DECIDE-AI  \u00b7  EU\u202fAI\u202fAct'
        '  \u00b7  FDA\u202fGMLP  \u00b7  ISO\u202f14971',
        ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color=MGRY)

# ── Save ──────────────────────────────────────────────────────────────────────
outpath = (r"D:\PhD\Manuscript\Manuscript\DigiB_Beyond-Accuracy"
           r"\DigitalBiomarkers_Submission\graphical-abstract.jpg")
fig.savefig(outpath, dpi=DPI, format='jpg', facecolor=BG,
            bbox_inches='tight', pad_inches=0.15)
plt.close()

from PIL import Image
im = Image.open(outpath)
print(f'Saved: {outpath}')
print(f'Size: {im.size[0]} x {im.size[1]} px  (max 4800 x 2100)')
print(f'DPI: {im.info.get("dpi", "not set")}')
print('OK' if im.size[0] <= 4800 and im.size[1] <= 2100
      else 'WARNING: exceeds Karger max size')
