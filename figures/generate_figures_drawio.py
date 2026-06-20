#!/usr/bin/env python3
"""
generate_figures_drawio.py
Convert Figs 1, 2, 3, 6 (diagram figures) to draw.io XML.

Layout source: generate_figures.py v4.0
All 4 figures → 4 pages in a single .drawio file.

Coordinate mapping per figure:
  px(x)  = x * sx
  dy(y)  = (y_max - y) * sy   (Y inverted: mpl origin bottom, draw.io top)
  box topleft = (cx - w/2)*sx, (y_max - cy - h/2)*sy
"""

# ── Colour palette (exact match to generate_figures.py) ─────────────────────
BD  = '#1C4E80'; BM  = '#2E7BBF'; BL  = '#D6E4F5'
GD  = '#007040'; GL  = '#D1F0E0'
AM  = '#C06800'; AL  = '#FFF0D0'
GRD = '#2D2D2D'; GRM = '#666666'; GRL = '#D4D4D4'
WH  = '#FFFFFF'; NBL = '#EEF2F6'   # neutral blue-grey (fig2 stage 3)

# ── XML escape ───────────────────────────────────────────────────────────────
def _esc(text):
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace('\n', '&lt;br/&gt;'))


# ── Page / diagram builder ───────────────────────────────────────────────────
class Page:
    """One diagram page in the .drawio file."""

    def __init__(self, name, pid, x_max, y_max, sx, sy=None):
        self.name  = name
        self.pid   = pid
        self.x_max = x_max
        self.y_max = y_max
        self.sx    = sx
        self.sy    = sy if sy is not None else sx
        self._cells = []
        self._id    = 2

    # ── coordinate helpers ───────────────────────────────────────────────────
    def px(self, x):       return x * self.sx
    def dy(self, y):       return (self.y_max - y) * self.sy
    def _bx(self, cx, w):  return (cx - w / 2) * self.sx
    def _by(self, cy, h):  return (self.y_max - cy - h / 2) * self.sy
    def _bw(self, w):      return w * self.sx
    def _bh(self, h):      return h * self.sy

    def _next(self):
        cid = self._id; self._id += 1; return str(cid)

    def _cell(self, value, style, vertex=True, x=None, y=None, w=None, h=None,
               src_x=None, src_y=None, tgt_x=None, tgt_y=None, pts=None):
        cid = self._next()
        if vertex:
            self._cells.append(
                f'        <mxCell id="{cid}" value="{_esc(value)}" '
                f'style="{style}" vertex="1" parent="1">\n'
                f'          <mxGeometry x="{x:.1f}" y="{y:.1f}" '
                f'width="{w:.1f}" height="{h:.1f}" as="geometry" />\n'
                f'        </mxCell>'
            )
        else:  # edge
            pts_xml = ''
            if pts:
                inner = ''.join(f'              <mxPoint x="{p[0]:.1f}" y="{p[1]:.1f}" />\n' for p in pts)
                pts_xml = f'            <Array as="points">\n{inner}            </Array>\n'
            self._cells.append(
                f'        <mxCell id="{cid}" value="" style="{style}" '
                f'edge="1" parent="1">\n'
                f'          <mxGeometry relative="1" as="geometry">\n'
                f'            <mxPoint x="{src_x:.1f}" y="{src_y:.1f}" as="sourcePoint" />\n'
                f'            <mxPoint x="{tgt_x:.1f}" y="{tgt_y:.1f}" as="targetPoint" />\n'
                f'{pts_xml}'
                f'          </mxGeometry>\n'
                f'        </mxCell>'
            )
        return cid

    # ── element methods ──────────────────────────────────────────────────────

    def rect(self, cx, cy, w, h, fc, ec, text, fs=9, bold=False, tc=None,
             lw=1.5, arc=8):
        tc = tc or ec
        fw = 1 if bold else 0
        style = (f'rounded=1;arcSize={arc};whiteSpace=wrap;html=1;'
                 f'fillColor={fc};strokeColor={ec};fontColor={tc};'
                 f'fontSize={fs};fontStyle={fw};'
                 f'align=center;verticalAlign=middle;strokeWidth={lw:.1f};')
        return self._cell(text, style, x=self._bx(cx, w), y=self._by(cy, h),
                          w=self._bw(w), h=self._bh(h))

    def dashrect(self, cx, cy, w, h, fc, ec, opacity=6, lw=1.0):
        """Dashed-border background rectangle (simulation zone, etc.)."""
        style = (f'rounded=1;arcSize=3;whiteSpace=wrap;html=1;'
                 f'fillColor={fc};strokeColor={ec};'
                 f'opacity={opacity};strokeWidth={lw:.1f};dashed=1;')
        return self._cell('', style, x=self._bx(cx, w), y=self._by(cy, h),
                          w=self._bw(w), h=self._bh(h))

    def label(self, cx, cy, w, h, text, fc=BD, fs=9, bold=False, italic=False):
        fw = (1 if bold else 0) | (2 if italic else 0)
        style = (f'text;html=1;strokeColor=none;fillColor=none;'
                 f'align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;'
                 f'fontColor={fc};fontSize={fs};fontStyle={fw};')
        return self._cell(text, style, x=self._bx(cx, w), y=self._by(cy, h),
                          w=self._bw(w), h=self._bh(h))

    def badge(self, cx, cy, r, num, bg=BD, fg=WH, fs=7.5):
        """Numbered circle badge (ellipse in draw.io)."""
        style = (f'ellipse;whiteSpace=wrap;html=1;'
                 f'fillColor={bg};strokeColor={fg};fontColor={fg};'
                 f'fontSize={fs:.1f};fontStyle=1;align=center;verticalAlign=middle;')
        x = self.px(cx) - self.sx * r
        y = self.dy(cy) - self.sy * r
        w = self.sx * r * 2;  h = self.sy * r * 2
        return self._cell(str(num), style, x=x, y=y, w=w, h=h)

    def hline(self, x1, x2, y, color=GRL, lw=1.0, dashed=False):
        d = 'dashed=1;' if dashed else ''
        style = f'endArrow=none;startArrow=none;strokeColor={color};strokeWidth={lw:.1f};{d}'
        return self._cell('', style, vertex=False,
                          src_x=self.px(x1), src_y=self.dy(y),
                          tgt_x=self.px(x2), tgt_y=self.dy(y))

    def vline(self, x, y_top, y_bot, color=GRL, lw=1.0, dashed=False):
        d = 'dashed=1;' if dashed else ''
        style = f'endArrow=none;startArrow=none;strokeColor={color};strokeWidth={lw:.1f};{d}'
        return self._cell('', style, vertex=False,
                          src_x=self.px(x), src_y=self.dy(y_top),
                          tgt_x=self.px(x), tgt_y=self.dy(y_bot))

    def harrow(self, x1, x2, y, color=GRD, lw=1.5):
        style = (f'endArrow=block;endFill=1;startArrow=none;'
                 f'strokeColor={color};strokeWidth={lw:.1f};edgeStyle=none;')
        return self._cell('', style, vertex=False,
                          src_x=self.px(x1), src_y=self.dy(y),
                          tgt_x=self.px(x2), tgt_y=self.dy(y))

    def varrow(self, x, y_from, y_to, color=GRD, lw=1.5):
        """Vertical arrow — y_from and y_to in matplotlib coordinates."""
        style = (f'endArrow=block;endFill=1;startArrow=none;'
                 f'strokeColor={color};strokeWidth={lw:.1f};edgeStyle=none;')
        return self._cell('', style, vertex=False,
                          src_x=self.px(x), src_y=self.dy(y_from),
                          tgt_x=self.px(x), tgt_y=self.dy(y_to))

    def darrow(self, x1, y1, x2, y2, color=GRD, lw=1.5):
        style = (f'endArrow=block;endFill=1;startArrow=none;'
                 f'strokeColor={color};strokeWidth={lw:.1f};edgeStyle=none;')
        return self._cell('', style, vertex=False,
                          src_x=self.px(x1), src_y=self.dy(y1),
                          tgt_x=self.px(x2), tgt_y=self.dy(y2))

    # ── XML assembly ─────────────────────────────────────────────────────────
    def xml(self, pw, ph):
        cells = '\n'.join(self._cells)
        return (
            f'  <diagram name="{self.name}" id="{self.pid}">\n'
            f'    <mxGraphModel dx="1200" dy="800" grid="0" gridSize="10" guides="1"\n'
            f'                  tooltips="1" connect="1" arrows="1" fold="0" page="1"\n'
            f'                  pageScale="1" pageWidth="{pw}" pageHeight="{ph}"\n'
            f'                  math="0" shadow="0" background="#FAFCFF">\n'
            f'      <root>\n'
            f'        <mxCell id="0" />\n'
            f'        <mxCell id="1" parent="0" />\n'
            f'{cells}\n'
            f'      </root>\n'
            f'    </mxGraphModel>\n'
            f'  </diagram>'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Conceptual Framework  (v4.0 layout)
# xlim 0–10, ylim 0–5.0, scale=100px/unit → 1000×500
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    p = Page('Fig 1 — Conceptual Framework', 'fig1', x_max=10, y_max=5.0, sx=100)

    # ── Column background bands (low-opacity fills) ───────────────────────────
    p.dashrect(1.85, 2.44, 3.30, 4.32, BL, BL, opacity=7)   # left column band
    p.dashrect(5.87, 2.47, 2.84, 4.32, BM, BM, opacity=4)   # centre column
    p.dashrect(8.82, 2.47, 2.40, 4.32, GD, GD, opacity=5)   # right column

    # ── Horizontal rule at y=4.18 ─────────────────────────────────────────────
    p.hline(0.20, 9.95, 4.18, color=GRL, lw=0.7)

    # ── Column headers ────────────────────────────────────────────────────────
    p.label(1.75, 4.38, 3.00, 0.36, 'Domain Pillars',     fc=BD, fs=8, bold=True, italic=True)
    p.label(5.85, 4.38, 2.76, 0.36, 'Governance Paradigm', fc=BD, fs=8, bold=True, italic=True)
    p.label(8.85, 4.38, 2.18, 0.36, 'Framework Output',   fc=GD, fs=8, bold=True, italic=True)

    # ── Three domain pillars (landscape: w=3.00, h=0.72) ─────────────────────
    pillars = [
        (3.42, ['Structured Clinical Reasoning', '(TiTrATE methodology)'], BL, BD),
        (2.47, ['Urgency Stratification', '(triage tiers: L / M / H)'],    GL, GD),
        (1.52, ['Explainability-aware Design', '(XAI as audit artefact)'], AL, AM),
    ]
    for cy, lines, fc, ec in pillars:
        p.rect(1.75, cy, 3.00, 0.72, fc, ec, '\n'.join(lines), fs=8, tc=GRD)

    # ── Merge bar: stubs → vertical bar → main arrow ──────────────────────────
    x_stub_end, x_trix_left = 3.60, 4.47
    for cy, *_ in pillars:
        p.harrow(3.25, x_stub_end, cy, color=GRM, lw=1.8)         # stub arrows
    p.vline(x_stub_end, 3.42, 1.52, color=GRM, lw=2.4)            # vertical bar
    p.harrow(x_stub_end, x_trix_left, 2.47, color=BD, lw=2.8)     # main merge arrow

    # ── TRI-X centre box (landscape: h=1.90) ─────────────────────────────────
    p.rect(5.85, 2.47, 2.76, 1.90, BD, BD,
           'TRI-X\nDecision-Governance\nProgramme',
           fs=11, bold=True, tc=WH)
    # Sub-label inside TRI-X (italic, light blue, near bottom of box)
    p.label(5.85, 1.79, 2.50, 0.26,
            'Uncertainty as Control Signal (escalation · abstention)',
            fc='#C0D8F0', fs=6.5, italic=True)

    # ── Arrow: TRI-X → Beyond Accuracy ───────────────────────────────────────
    p.harrow(7.23, 7.76, 2.47, color=GD, lw=2.8)

    # ── Beyond Accuracy output box (landscape: h=1.90) ────────────────────────
    p.rect(8.85, 2.47, 2.18, 1.90, GD, GD,
           'Simulation-based\nSafety Evaluation\n(Beyond Accuracy)',
           fs=9, bold=True, tc=WH)

    return p, 1000, 540


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Evaluation Pipeline  (v4.0 layout)
# xlim 0–10, ylim 0–3.2, scale=100px/unit → 1000×320
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig3():
    p = Page('Fig 3 — Evaluation Pipeline', 'fig3', x_max=10, y_max=3.2, sx=100)

    bw, bh, cy = 1.40, 1.10, 1.75
    x0, step   = 0.76, 1.68   # stage centres

    stages = [
        (0, ['Validated', 'Archetypes', '+ Risk Tiers', '(L / M / H)'],
         BL, BD, GRD, BD, 'Input'),
        (1, ['Scenario', 'Instantiation', '(mask · noise ·', 'conflict · degrade)'],
         AL, AM, GRD, AM, 'Perturbation'),
        (2, ['System', 'Under Test', '(CDSS)'],
         NBL, GRM, GRD, GRM, 'Inference'),
        (3, ['Log Auditable', 'Artefacts', '(versioned;', 'seeded)'],
         BL, BD, GRD, BD, 'Logging'),
        (4, ['Compute Metrics', 'ECE, AURC,', 'L_harm, η_xai,', 'Ĉ_α'],
         BL, BD, GRD, BD, 'Metrics'),
        (5, ['Governance', 'Verdict', 'Pre-deployment', 'Evidence'],
         GD, GD, WH, GD, 'Verdict'),
    ]

    for i, lines, fc, ec, tc, badge_c, phase in stages:
        cx = x0 + i * step
        p.rect(cx, cy, bw, bh, fc, ec, '\n'.join(lines), fs=7, tc=tc)
        p.badge(cx, cy + bh/2 + 0.20, 0.18, i + 1, bg=badge_c)
        p.label(cx, 2.95, bw, 0.22, phase, fc=badge_c, fs=7, bold=True, italic=True)
        if i < len(stages) - 1:
            nx = x0 + (i + 1) * step
            p.harrow(cx + bw/2 + 0.02, nx - bw/2 - 0.02, cy, color=GRD, lw=2.0)

    return p, 1000, 360


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Archetype-to-Scenario Pipeline  (v4.0 layout)
# xlim 0–8.8, ylim 0–2.7, scale=100px/unit → 880×270
# Stage 4 fixed: GL/GD/GRD (was GD/GD/WH) — consistent light bg + dark text
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig2():
    p = Page('Fig 2 — Archetype Pipeline', 'fig2', x_max=8.8, y_max=2.7, sx=100)

    bw, bh, cy = 1.72, 1.10, 1.42
    x0, step   = 0.92, 2.00

    phases = ['Archetype', 'Perturbation', 'Instantiation', 'Evaluation']

    stages = [
        (0, ['Clinical Archetype', '(abstract + tier', 'r ∈ {L, M, H})'],
         BL, BD, GRD, BD),
        (1, ['Perturbation', 'Operators', 'p_mask, p_noise,', 'p_conflict, p_degrade'],
         AL, AM, GRD, AM),
        (2, ['Scenario Set', '(Nₐ instances;', 'varied uncertainty', 'profile uᵢ)'],
         GL, GD, GRD, GD),
        (3, ['Safety Metrics', 'ECE(r), AURC,', 'L_harm, η_xai,', 'Ĉ_α'],
         GL, GD, GRD, GD),   # Fixed: was GD/GD/WH
    ]

    for i, lines, fc, ec, tc, badge_c in stages:
        cx = x0 + i * step
        p.rect(cx, cy, bw, bh, fc, ec, '\n'.join(lines), fs=7, tc=tc)
        p.badge(cx, cy + bh/2 + 0.18, 0.17, i + 1, bg=badge_c)
        p.label(cx, 2.55, bw, 0.22, phases[i], fc=badge_c, fs=7, bold=True, italic=True)
        if i < len(stages) - 1:
            nx = x0 + (i + 1) * step
            p.harrow(cx + bw/2 + 0.02, nx - bw/2 - 0.02, cy, color=GRD, lw=2.0)

    return p, 880, 310


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — V3 Lifecycle Positioning
# xlim 0–1, ylim 0–1, sx=1000, sy=500  (physical H = W×0.50)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig6():
    p = Page('Fig 6 — V3 Lifecycle', 'fig6', x_max=1.0, y_max=1.0, sx=1000, sy=500)

    cy, bh = 0.46, 0.30
    bot     = cy - bh / 2   # = 0.31

    # ── Three lifecycle stage boxes ───────────────────────────────────────────
    stage_data = [
        ('Verification',           0.08, 0.22, BL, BD),
        ('Analytical\nValidation', 0.40, 0.26, GL, GD),
        ('Clinical\nValidation',   0.74, 0.22, AL, AM),
    ]
    for label, cx, bw, fc, ec in stage_data:
        p.rect(cx, cy, bw, bh, fc, ec, label, fs=9, bold=True, tc=ec)

    # ── Stage arrows ──────────────────────────────────────────────────────────
    # V → AV: right edge of V = 0.08+0.11=0.19, left edge of AV = 0.40-0.13=0.27
    p.harrow(0.19 + 0.02, 0.27 - 0.02, cy, color=GRD, lw=2.0)
    # AV → CV: right edge of AV = 0.53, left edge of CV = 0.63
    p.harrow(0.53 + 0.02, 0.63 - 0.02, cy, color=GRD, lw=2.0)

    # ── Simulation zone dashed background ────────────────────────────────────
    # Covers AV box + margins: x=0.24 to 0.56, y=0.14 to 0.90
    p.dashrect(0.40, 0.52, 0.32, 0.76, GD, GD, opacity=6, lw=1.0)

    # ── Zone label at top ─────────────────────────────────────────────────────
    p.label(0.40, 0.94, 0.70, 0.06,
            'Simulation-based evaluation zone  (no patient-level data required)',
            fc=GD, fs=7, italic=True)

    # ── "THIS FRAMEWORK" annotation + down arrow ──────────────────────────────
    p.label(0.40, 0.86, 0.30, 0.06, 'THIS FRAMEWORK', fc=GD, fs=8, bold=True)
    # Arrow from just below label text down to top of AV box (cy + bh/2 = 0.61)
    p.varrow(0.40, 0.83, cy + bh/2 + 0.01, color=GD, lw=1.8)

    # ── Sub-labels inside Verification ───────────────────────────────────────
    p.label(0.08, cy - 0.10, 0.20, 0.10,
            'Sensor specs\nDevice testing', fc=GRM, fs=7)

    # ── Sub-labels inside Analytical Validation (inner items) ────────────────
    inner = [
        (cy - 0.03 - 0 * 0.050, '𝒫: 4 perturbation operators'),
        (cy - 0.03 - 1 * 0.050, 'ℳ: 5-metric safety suite'),
        (cy - 0.03 - 2 * 0.050, 'Φ: governance verdict'),
    ]
    for iy, text in inner:
        p.label(0.40, iy, 0.24, 0.05, text, fc=GD, fs=7)

    # ── Sub-labels inside Clinical Validation ────────────────────────────────
    p.label(0.74, cy - 0.10, 0.20, 0.10,
            'Real patient data\nProspective trials', fc=GRM, fs=7)

    # ── Governance artefact labels below boxes (dashed arrows up to box gaps) ─
    #    mid1 = midpoint between V right and AV left
    mid1 = (0.08 + 0.11 + 0.02 + 0.40 - 0.13 - 0.02) / 2   # 0.23
    mid2 = (0.40 + 0.13 + 0.02 + 0.74 - 0.11 - 0.02) / 2   # 0.58

    p.label(mid1, 0.10, 0.16, 0.08, 'Archetype\nspec', fc=BD, fs=6.5)
    p.darrow(mid1, 0.14, mid1, bot - 0.01, color=BD, lw=0.9)   # upward arrow

    p.label(mid2, 0.10, 0.16, 0.08, 'Protocol\ncard', fc=AM, fs=6.5)
    p.darrow(mid2, 0.14, mid2, bot - 0.01, color=AM, lw=0.9)   # upward arrow

    # ── Bottom attribution ────────────────────────────────────────────────────
    p.label(0.50, 0.02, 0.90, 0.06,
            'V3 Digital Biomarker Development Lifecycle (Goldsack et al. 2020)',
            fc=GRM, fs=7.5, italic=True)

    return p, 1000, 540


# ═══════════════════════════════════════════════════════════════════════════════
# Main — assemble all 4 pages into one .drawio file
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    pages = [make_fig1(), make_fig2(), make_fig3(), make_fig6()]  # Fig2=archetype, Fig3=pipeline

    import os
    here = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(here, 'beyond-accuracy-figures.drawio')

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<mxfile host="app.diagrams.net" version="21.7.5">\n')
        for page, pw, ph in pages:
            f.write(page.xml(pw, ph) + '\n')
        f.write('</mxfile>\n')

    # Validate
    import xml.etree.ElementTree as ET
    tree  = ET.parse(outpath)
    cells = tree.findall('.//mxCell')
    pages_xml = tree.findall('.//diagram')
    import os
    sz = os.path.getsize(outpath)

    print(f'Saved:   {outpath}')
    print(f'Pages:   {len(pages_xml)}  ({", ".join(d.get("name") for d in pages_xml)})')
    print(f'Cells:   {len(cells)} mxCell elements across all pages')
    print(f'Size:    {sz:,} bytes  ({sz//1024} KB)')
