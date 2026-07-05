"""Render the methodology pipeline as a flowchart (presentables/methodology_flowchart.png).

Deliberately coarse-grained: shows pipeline *stages*, not every embedding family as its
own box (that's what RESULTS.md's tables are for). The one place this diagram uses
color is the embedding-family taxonomy, because that grouping is the actual finding of
the study -- everything else is a single neutral tone.

Usage: python src/flowchart.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'presentables')
os.makedirs(OUT_DIR, exist_ok=True)

STAGE = '#3d4f66'    # single neutral tone for every non-taxonomy box
STAGE_DARK = '#232f3d'  # slightly darker, for the final/summary box only
# same 4 colors used in figures.py/GROUP_COLOR, reused here for one-to-one consistency
CAT_COLOR = {
    'identity': '#c1440e', 'nonidentity': '#4a7c59',
    'classical_identity': '#8b5a2b', 'handcrafted': '#7a7a7a',
}


def box(ax, cx, cy, w, h, text, color, fontsize=10, text_color='white', ha='center'):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle='round,pad=0.02,rounding_size=0.06',
                        linewidth=1.0, edgecolor='#1a1a1a', facecolor=color, zorder=2)
    ax.add_patch(b)
    tx = cx if ha == 'center' else cx - w / 2 + 0.15
    ax.text(tx, cy, text, ha=ha, va='center', fontsize=fontsize,
            color=text_color, zorder=3, linespacing=1.4)
    return (cx, cy, w, h)


def _edge(b, side):
    x, y, w, h = b
    return {'bottom': (x, y - h / 2), 'top': (x, y + h / 2),
            'left': (x - w / 2, y), 'right': (x + w / 2, y)}[side]


def arrow(ax, b1, b2, start='bottom', end='top', label=None, color='#555555'):
    p1, p2 = _edge(b1, start), _edge(b2, end)
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=14,
                        linewidth=1.2, color=color, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=8, style='italic', color='#333333', va='center')


def side_arrow(ax, b1, side1, b2, side2, via_x, color='#555555'):
    """Route a connector out along a side margin (a clean 'staple' shape) so it
    doesn't cut across boxes sitting between b1 and b2."""
    p1, p2 = _edge(b1, side1), _edge(b2, side2)
    ax.plot([p1[0], via_x], [p1[1], p1[1]], color=color, linewidth=1.2, zorder=1)
    ax.plot([via_x, via_x], [p1[1], p2[1]], color=color, linewidth=1.2, zorder=1)
    a = FancyArrowPatch((via_x, p2[1]), p2, arrowstyle='-|>', mutation_scale=14,
                        linewidth=1.2, color=color, zorder=1)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(10.5, 13.0))
    ax.set_xlim(-5.6, 5.6)
    ax.set_ylim(1.3, 22.3)
    ax.axis('off')
    ax.set_title('Methodology: from raw photo to invariance/accuracy correlation',
                 fontsize=14, fontweight='bold', pad=12)

    # ---- datasets ----
    y = 21.2
    d1 = box(ax, -3.4, y, 3.2, 1.3, 'SCUT-FBP5500\n5,500 images, scale 1–5', STAGE, fontsize=9)
    d2 = box(ax, 0, y, 3.2, 1.3, 'MEBeauty\n2,197 images, scale ~1–10', STAGE, fontsize=9)
    d3 = box(ax, 3.4, y, 3.2, 1.3, 'London Set\n102 IDs × 10 views, scale 1–7', STAGE, fontsize=9)

    # ---- manifest ----
    y2 = 18.9
    man = box(ax, 0, y2, 7.0, 1.1, 'unified manifest.csv\nimage_id, path, score, gender, ethnicity, age, dataset, view', STAGE, fontsize=9.5)
    for d in (d1, d2, d3):
        arrow(ax, d, man)

    # ---- preprocessing (detection + alignment + crop, one box) ----
    y3 = 16.6
    prep = box(ax, 0, y3, 7.6, 1.5,
               'Preprocessing\nFace detection + landmark alignment (RetinaFace)\ncropped to what each family expects',
               STAGE, fontsize=9.5)
    arrow(ax, man, prep)

    # ---- embedding families: ONE box, taxonomy shown as an inset 2x2 ----
    # Layout is computed top-down from the box's own top edge so the chip grid is
    # guaranteed to fit inside outer_h with equal margins (verified: bottom chip edge
    # sits 0.13 above outer's bottom edge, matching the 0.15 top text margin).
    y5 = 13.0
    outer_w, outer_h = 10.6, 3.6
    outer = box(ax, 0, y5, outer_w, outer_h, '', STAGE, fontsize=9)
    top = y5 + outer_h / 2
    ax.text(0, top - 0.35, '11 frozen, pretrained embedding families',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=3)
    ax.text(0, top - 0.75, 'no fine-tuning — identical downstream probe applied to every one',
            ha='center', va='center', fontsize=8, style='italic', color='#d8dee8', zorder=3)

    chip_w, chip_h, chip_gap, gx = 4.9, 1.15, 0.12, 0.15
    col_x = [-chip_w / 2 - gx / 2, chip_w / 2 + gx / 2]
    grid_top = top - 1.05  # below the title/subtitle block
    row_y = [grid_top - chip_h / 2, grid_top - chip_h - chip_gap - chip_h / 2]
    grid = [
        (col_x[0], row_y[0], 'Deep, trained for identity (4)', 'FaceNet · ArcFace · CosFace · AdaFace', CAT_COLOR['identity']),
        (col_x[1], row_y[0], 'Deep, not for identity (4)', 'FairFace · CLIP · DINOv2 · Blendshapes', CAT_COLOR['nonidentity']),
        (col_x[0], row_y[1], 'Classical, for identity (1)', 'Fisherface (PCA+LDA, fit on London IDs)', CAT_COLOR['classical_identity']),
        (col_x[1], row_y[1], 'Classical, hand-crafted (2)', 'Geometric · LBPH', CAT_COLOR['handcrafted']),
    ]
    for cx_, cy_, title, sub, col in grid:
        box(ax, cx_, cy_, chip_w, chip_h, f'{title}\n{sub}', col, fontsize=7.8)

    arrow(ax, prep, outer, start='bottom', end='top')

    # ---- ridge probe ----
    y6 = 9.6
    probe = box(ax, 0, y6, 8.6, 1.2,
                'Identical ridge-regression probe, per (family, dataset)\nStandardScaler → Ridge (α via inner 3-fold CV)',
                STAGE, fontsize=9.5)
    arrow(ax, outer, probe, start='bottom', end='top')

    # ---- two evaluation branches ----
    y7 = 7.3
    within = box(ax, -2.0, y7, 4.5, 1.4, 'Within-dataset accuracy\n10-fold CV', STAGE, fontsize=9)
    cross = box(ax, 2.0, y7, 4.5, 1.4, 'Cross-dataset accuracy\ntrain A → test B, 6 pairs', STAGE, fontsize=9)
    for b in (within, cross):
        arrow(ax, probe, b, start='bottom', end='top')

    # ---- pose invariance (fed directly by the embedding stage) ----
    y8 = 4.9
    inv = box(ax, 0, y8, 8.6, 1.4,
              "Pose-invariance d′ (London's 10 views/identity)\nsame-identity cross-pose similarity vs. different-identity frontal similarity",
              STAGE, fontsize=9)
    side_arrow(ax, outer, 'left', inv, 'left', via_x=-5.45)

    # ---- final correlation ----
    y9 = 2.5
    final = box(ax, 0, y9, 9.4, 1.4,
                "Correlate d′ vs. accuracy: pooled (9 trained families)\n+ by-category (4 groups) — results/invariance.csv, category_summary.csv",
                STAGE_DARK, fontsize=9)
    arrow(ax, inv, final)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'methodology_flowchart.png')
    fig.savefig(out, dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    main()
