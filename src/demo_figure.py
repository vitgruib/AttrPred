"""One-photo illustration: what each of the 11 embedding families predicts for a
single face, using the same ridge probes trained in evaluate.py (fit on SCUT).

The photo (data/demo/demo_photo.jpg) is a CC0/free-to-use Unsplash portrait,
not part of any of the three rated datasets -- this is a portfolio
illustration of the pipeline in action, not a benchmark result (see
figures/accuracy_*.png for the real numbers). Regenerating the prediction
column for a different photo requires re-running src/demo_predict.py on that
photo; this script only draws the figure from tables/demo_predictions.csv.

Usage: python src/demo_figure.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from figures import GROUP_COLOR, GROUP_LABEL, DISPLAY_NAME, display, legend_handles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
RES = os.path.join(ROOT, 'figures')
DATA = os.path.join(ROOT, 'data', 'demo')


def main():
    rows = list(csv.DictReader(open(os.path.join(TAB, 'demo_predictions.csv'))))
    fams = sorted((r['family'] for r in rows), key=lambda f: -float(next(r['predicted_score'] for r in rows if r['family'] == f)))
    scores = {r['family']: float(r['predicted_score']) for r in rows}

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.05)

    ax_img = fig.add_subplot(gs[0])
    img = mpimg.imread(os.path.join(DATA, 'demo_photo.jpg'))
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title('input photo', fontsize=11)

    ax = fig.add_subplot(gs[1])
    y = np.arange(len(fams))
    vals = [scores[f] for f in fams]
    colors = [GROUP_COLOR[f] for f in fams]
    ax.barh(y, vals, color=colors)
    for i, f in enumerate(fams):
        ax.text(vals[i] + 0.03, i, f'{vals[i]:.2f}', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([display(f) for f in fams])
    ax.invert_yaxis()
    ax.set_xlim(1, 5)
    ax.set_xlabel('predicted attractiveness score (1-5 scale, ridge probe trained on SCUT-FBP5500)')
    ax.set_title("Same photo, 11 embeddings, 11 different guesses", fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles(), frameon=False, loc='lower right', fontsize=8)

    fig.tight_layout()
    out = os.path.join(RES, 'demo_predictions.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
