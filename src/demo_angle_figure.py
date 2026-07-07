"""Same person, three head angles (frontal / three-quarter / profile, three frames
from one continuous rotation video), same 11 SCUT-trained ridge probes. Illustrates
the pose-invariance / accuracy tradeoff from the main study directly: identity-trained
embeddings are built to give a near-constant representation across pose, so their
predicted score should barely move; embeddings that were never trained for pose
invariance are free to let appearance (and therefore the predicted score) drift
with the angle.

Usage: python src/demo_angle_figure.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from figures import GROUP_COLOR, GROUP_LABEL, display, legend_handles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
RES = os.path.join(ROOT, 'figures')
DATA = os.path.join(ROOT, 'data', 'demo')

ANGLES = ['frontal', 'threequarter', 'profile']
ANGLE_LABEL = {'frontal': 'Frontal', 'threequarter': 'Three-quarter', 'profile': 'Profile'}


def declutter(values, min_gap):
    """Nudge apart y-positions (for text labels) that would otherwise collide,
    keeping everything else about the plot untouched. values: {key: y}."""
    order = sorted(values, key=lambda k: -values[k])
    adjusted = dict(values)
    for prev, cur in zip(order, order[1:]):
        if adjusted[prev] - adjusted[cur] < min_gap:
            adjusted[cur] = adjusted[prev] - min_gap
    return adjusted


def main():
    results = json.load(open(os.path.join(TAB, 'demo_angle_predictions.json')))
    fams = sorted(results, key=lambda f: -(max(results[f].values()) - min(results[f].values())))

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.12)

    ax_img = fig.add_subplot(gs[0])
    imgs = [mpimg.imread(os.path.join(DATA, f'demo_angle_{a}.jpg')) for a in ANGLES]
    gap = np.full((imgs[0].shape[0], 10, 3), 255, dtype=imgs[0].dtype)
    row = np.concatenate([imgs[0], gap, imgs[1], gap, imgs[2]], axis=1)
    ax_img.imshow(row)
    ax_img.axis('off')

    ax = fig.add_subplot(gs[1])
    x = np.arange(len(ANGLES))
    all_y = [v for f in fams for v in results[f].values()]
    ymin, ymax = min(all_y), max(all_y)
    label_y = declutter({f: results[f][ANGLES[-1]] for f in fams}, (ymax - ymin) * 0.045)

    for f in fams:
        y = [results[f][a] for a in ANGLES]
        ax.plot(x, y, marker='o', color=GROUP_COLOR[f], linewidth=1.8, markersize=5, alpha=0.9)
        ax.plot([x[-1], x[-1] + 0.12], [y[-1], label_y[f]], color=GROUP_COLOR[f], linewidth=0.6, alpha=0.6)
        ax.annotate(display(f), (x[-1] + 0.15, label_y[f]), va='center', fontsize=8, color=GROUP_COLOR[f])

    ax.set_xticks(x)
    ax.set_xticklabels([ANGLE_LABEL[a] for a in ANGLES], fontsize=11)
    ax.set_xlim(-0.15, 2.75)
    ax.set_ylabel('predicted attractiveness score (1-5 scale, ridge probe trained on SCUT-FBP5500)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles(), frameon=False, loc='upper left', fontsize=8)

    fig.suptitle('Predicted attractiveness score by head angle, per embedding family', fontsize=13)
    fig.subplots_adjust(left=0.11, right=0.86, top=0.92, bottom=0.08)
    out = os.path.join(RES, 'demo_angle_predictions.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
