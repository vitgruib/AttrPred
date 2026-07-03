"""Generate summary figures from the existing results/*.csv (no new computation).

Outputs:
  results/accuracy_by_family.png          grouped bars: accuracy per family x dataset
  results/invariance_and_accuracy_bars.png two-panel: d' and accuracy, same order, aligned

Usage: python src/figures.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

DATASET_LABEL = {'scut': 'SCUT-FBP5500', 'mebeauty': 'MEBeauty', 'london': 'London'}
DATASETS = ['scut', 'mebeauty', 'london']

# color by what the embedding was trained to do
GROUP_COLOR = {
    'facenet': '#c1440e', 'arcface': '#c1440e', 'cosface': '#c1440e', 'adaface': '#c1440e',  # identity-trained
    'fairface': '#4a7c59', 'clip': '#4a7c59',                                                  # not identity-trained
    'geometric': '#7a7a7a',                                                                    # untrained / hand-crafted
}
GROUP_LABEL = {
    'facenet': 'trained for identity', 'arcface': 'trained for identity',
    'cosface': 'trained for identity', 'adaface': 'trained for identity',
    'fairface': 'trained, not for identity', 'clip': 'trained, not for identity',
    'geometric': 'hand-crafted (untrained)',
}


def legend_handles():
    seen, handles = set(), []
    for fam, lab in GROUP_LABEL.items():
        if lab not in seen:
            seen.add(lab)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[fam], label=lab))
    return handles


def fig_accuracy_by_family():
    acc = defaultdict(dict)  # family -> dataset -> pearson
    with open(os.path.join(RES, 'within.csv')) as f:
        for r in csv.DictReader(f):
            if r['probe'] == 'ridge':
                acc[r['family']][r['dataset']] = float(r['pearson'])

    families = sorted(acc, key=lambda fam: -np.mean(list(acc[fam].values())))
    x = np.arange(len(families))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, ds in enumerate(DATASETS):
        vals = [acc[fam].get(ds, 0) for fam in families]
        ax.bar(x + (i - 1) * width, vals, width, label=DATASET_LABEL[ds], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() if f != 'clip' else 'CLIP' for f in families], rotation=20, ha='right')
    ax.set_ylabel('Attractiveness prediction accuracy\n(Pearson r, ridge probe, 5-fold CV)')
    ax.set_title('Within-dataset accuracy by embedding family and dataset')
    ax.legend(title='Dataset', frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RES, 'accuracy_by_family.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


def fig_invariance_and_accuracy_bars():
    rows = list(csv.DictReader(open(os.path.join(RES, 'invariance.csv'))))
    rows.sort(key=lambda r: -float(r['dprime_pose']))
    fams = [r['family'] for r in rows]
    dp = [float(r['dprime_pose']) for r in rows]
    acc = [float(r['mean_within_r']) for r in rows]
    colors = [GROUP_COLOR[f] for f in fams]
    labels = [f.capitalize() if f != 'clip' else 'CLIP' for f in fams]
    x = np.arange(len(fams))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.bar(x, dp, color=colors)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_ylabel("d'  (pose-robust identity\ndiscriminability)")
    ax1.set_title("Pose invariance (top) vs. attractiveness accuracy (bottom)\n"
                  "sorted by invariance -- inverse pattern among trained models",
                  fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    ax2.bar(x, acc, color=colors)
    ax2.set_ylabel('mean within-dataset\naccuracy (Pearson r)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(handles=legend_handles(), frameon=False, loc='upper right', fontsize=9)

    fig.tight_layout()
    out = os.path.join(RES, 'invariance_and_accuracy_bars.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    fig_accuracy_by_family()
    fig_invariance_and_accuracy_bars()
