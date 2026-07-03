"""Render the methodology pipeline as a flowchart (presentables/methodology_flowchart.png).

Purely a visual summary of what src/preprocess.py, extract.py, evaluate.py,
invariance.py, and bias.py actually do -- no new computation.
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

COLOR = dict(
    data='#3b6ea5', detect='#c1440e', crop='#b58900',
    identity='#c1440e', nonidentity='#4a7c59', geometric='#7a7a7a',
    probe='#6a4c93', analysis='#2a6f77', decision='#8b0000',
)


def box(ax, cx, cy, w, h, text, color, fontsize=9.5, fontweight='normal', text_color='white'):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle='round,pad=0.02,rounding_size=0.08',
                        linewidth=1.2, edgecolor='black', facecolor=color, zorder=2)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
             fontweight=fontweight, color=text_color, zorder=3, linespacing=1.35)
    return (cx, cy, w, h)


def arrow(ax, b1, b2, start='bottom', end='top', label=None, style='-|>', color='black'):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {'bottom': (x1, y1 - h1 / 2), 'top': (x1, y1 + h1 / 2),
           'left': (x1 - w1 / 2, y1), 'right': (x1 + w1 / 2, y1)}
    p1 = pts[start] if start in ('bottom', 'top', 'left', 'right') else (x1, y1)
    pts2 = {'bottom': (x2, y2 - h2 / 2), 'top': (x2, y2 + h2 / 2),
            'left': (x2 - w2 / 2, y2), 'right': (x2 + w2 / 2, y2)}
    p2 = pts2[end]
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                        linewidth=1.3, color=color, zorder=1,
                        connectionstyle='arc3,rad=0.0')
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=7.5, style='italic', color='#333333', va='center')


def main():
    fig, ax = plt.subplots(figsize=(13.5, 19))
    ax.set_xlim(-0.7, 15.2)
    ax.set_ylim(7.5, 34)
    ax.axis('off')
    ax.set_title('Methodology pipeline: from raw photo to attractiveness-accuracy /\n'
                 'pose-invariance correlation', fontsize=14, fontweight='bold', pad=10)

    CX = 7.25  # figure midline, everything else is centered relative to this

    # ---- Row: raw datasets ----
    y = 32.5
    d1 = box(ax, CX - 5.0, y, 3.6, 1.6, 'SCUT-FBP5500\n5,500 images\nscale 1-5, 60 fixed raters', COLOR['data'])
    d2 = box(ax, CX, y, 3.6, 1.6, 'MEBeauty\n2,197 rated images\nscale ~1-10, ~300-rater pool', COLOR['data'])
    d3 = box(ax, CX + 5.0, y, 3.6, 1.6, 'London Set\n102 identities x 10 views\nscale 1-7, 2,513 raters (front only)', COLOR['data'])

    # ---- manifest ----
    y2 = 30.2
    man = box(ax, CX, y2, 6.5, 1.1, 'manifest.csv -- unified: image_id, path, score,\ngender, ethnicity, dataset, view, rated',
              '#274b6d', fontsize=9)
    for d in (d1, d2, d3):
        arrow(ax, d, man, start='bottom', end='top')

    # ---- detection ----
    y3 = 27.9
    det = box(ax, CX, y3, 7.5, 1.6,
             'Face detection + 5-pt landmarks\nRetinaFace (insightface buffalo_l)\nlargest face wins if multiple detected',
             COLOR['detect'])
    arrow(ax, man, det, start='bottom', end='top')

    # ---- retry decision ----
    y4 = 25.8
    retry = box(ax, CX, y4, 9.0, 1.9,
               'no face found?\n-> reflect-pad border (0.4x) + retry\nstill none & image small? -> upscale 2x, pad, retry\n'
               'final failures: 1 / 8,718 images',
               COLOR['decision'], fontsize=8.5)
    arrow(ax, det, retry, start='bottom', end='top', label='on failure')

    # ---- two crop outputs ----
    y5 = 23.2
    c112 = box(ax, CX - 3.0, y5, 4.8, 1.5,
              'Aligned 112x112 crop\n(5-pt similarity transform,\nArcFace template)', COLOR['crop'])
    c224 = box(ax, CX + 3.0, y5, 4.8, 1.5,
              'Loose 224x224 crop\n(1.3x bbox expansion)', COLOR['crop'])
    arrow(ax, retry, c112, start='bottom', end='top')
    arrow(ax, retry, c224, start='bottom', end='top')

    # ---- embedding extractors (7 boxes, evenly spaced, gap between the two groups) ----
    y6 = 20.4
    fam_h = 1.55
    fam_w = 1.95
    gap = 0.25
    group_gap = 0.65
    identity_fams = [('FaceNet', 'InceptionResnetV1\ntriplet loss, VGGFace2\n512-d'),
                     ('ArcFace', 'ResNet-50\nangular margin, WebFace600K\n512-d'),
                     ('CosFace', 'ResNet-100\ncosine margin m=0.4, Glint360K\n512-d'),
                     ('AdaFace', 'IR-101\nadaptive margin, WebFace4M\n512-d')]
    other_fams = [('FairFace', 'ResNet-34 (penult.)\nattribute classifier\n512-d', COLOR['nonidentity']),
                 ('CLIP', 'ViT-L/14\nimage-text contrastive\n768-d', COLOR['nonidentity']),
                 ('Geometric', '106-pt landmarks (on\nORIGINAL image) -> ratios\n221-d', COLOR['geometric'])]
    total_w = 4 * fam_w + 3 * gap + group_gap + 3 * fam_w + 2 * gap
    left = CX - total_w / 2
    xs4, x = [], left + fam_w / 2
    for _ in identity_fams:
        xs4.append(x)
        x += fam_w + gap
    x += group_gap - gap
    xs3 = []
    for _ in other_fams:
        xs3.append(x)
        x += fam_w + gap

    fam_boxes = {}
    for x_, (name, sub) in zip(xs4, identity_fams):
        b = box(ax, x_, y6, fam_w, fam_h, f'{name}\n{sub}', COLOR['identity'], fontsize=6.8)
        arrow(ax, c112, b, start='bottom', end='top')
        fam_boxes[name] = b
    for x_, (name, sub, col) in zip(xs3, other_fams):
        b = box(ax, x_, y6, fam_w, fam_h, f'{name}\n{sub}', col, fontsize=6.8)
        src = c224 if name != 'Geometric' else retry  # geometric re-detects on the ORIGINAL image
        arrow(ax, src, b, start='bottom' if name != 'Geometric' else 'right',
              end='top' if name != 'Geometric' else 'right')
        fam_boxes[name] = b

    # ---- convergence: identical ridge probe ----
    y7 = 17.6
    probe = box(ax, CX, y7, 10.4, 1.3,
               'Identical ridge-regression probe per (family, dataset)\n'
               'StandardScaler -> Ridge(alpha tuned by inner 3-fold CV)', COLOR['probe'])
    for name in fam_boxes:
        arrow(ax, fam_boxes[name], probe, start='bottom', end='top')

    # ---- three evaluation branches ----
    y8 = 15.1
    within = box(ax, CX - 4.3, y8, 4.0, 1.5,
                'Within-dataset:\n5-fold CV -> Pearson r,\nSpearman rho, RMSE', COLOR['analysis'])
    cross = box(ax, CX, y8, 4.0, 1.5,
               'Cross-dataset:\ntrain on A (z-scored) ->\ntest on B, 6 ordered pairs', COLOR['analysis'])
    bias = box(ax, CX + 4.3, y8, 4.0, 1.5,
              'Bias breakdown:\nsame CV predictions, split\nby gender / ethnicity subgroup', COLOR['analysis'])
    for b in (within, cross, bias):
        arrow(ax, probe, b, start='bottom', end='top')

    # ---- pose-invariance branch (from London embeddings directly) ----
    y9 = 12.3
    inv = box(ax, CX, y9, 8.6, 1.7,
             "Pose invariance d' (London's 10 views/identity only)\n"
             "L2-normalize -> same-identity cross-pose similarity vs.\n"
             "different-identity frontal similarity, signal-detection d'",
             '#1b4f4f', fontsize=8.5)
    arrow(ax, fam_boxes['FaceNet'], inv, start='left', end='top', color='#555555')
    ax.text(CX, y9 + 1.05, "each of the 7 families' London embeddings feed this stage",
           fontsize=7.5, ha='center', style='italic', color='#333333')

    # ---- final correlation ----
    y10 = 9.4
    final = box(ax, CX, y10, 9.4, 1.6,
               "Correlate per-family d' vs. mean within/cross accuracy\n"
               "(6 trained families; hand-crafted Geometric reported separately)\n"
               "-> results/invariance.csv, presentables figures", '#3a2a5c', fontsize=8.5)
    arrow(ax, within, final, start='bottom', end='top', color='#888888')
    arrow(ax, cross, final, start='bottom', end='top', color='#888888')
    arrow(ax, inv, final, start='bottom', end='top')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'methodology_flowchart.png')
    fig.savefig(out, dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    main()
