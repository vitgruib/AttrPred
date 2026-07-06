"""Generate summary figures from the existing tables/*.csv (no new computation).

tables/ holds raw data (CSVs) only; figures are written to figures/ so each
figure has exactly one copy in the repo.

Outputs:
  figures/accuracy_by_family.png          grouped bars: accuracy per family x dataset
  figures/invariance_and_accuracy_bars.png two-panel: d' and accuracy, same order, aligned
  figures/bias_gap_by_family.png          two-panel: ethnicity gap and gender gap, same order
  figures/bias_ethnicity_heatmap.png      family x ethnicity-group accuracy grid (MEBeauty)
  figures/accuracy_aggregate_by_family.png single bar per family, mean accuracy across all 3 datasets
  figures/bias_aggregate_by_ethnicity.png  single bar per ethnicity group, pooled across all families/datasets

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
TAB = os.path.join(ROOT, 'tables')
RES = os.path.join(ROOT, 'figures')

DATASET_LABEL = {'scut': 'SCUT-FBP5500', 'mebeauty': 'MEBeauty', 'london': 'London'}
DATASETS = ['scut', 'mebeauty', 'london']

# color by what the embedding was trained to do (deep vs. classical x identity vs. not)
GROUP_COLOR = {
    'facenet': '#c1440e', 'arcface': '#c1440e', 'cosface': '#c1440e', 'adaface': '#c1440e',  # deep, identity-trained
    'fairface': '#4a7c59', 'clip': '#4a7c59', 'dinov2': '#4a7c59', 'blendshapes': '#4a7c59',  # deep, not identity-trained
    'fisherface': '#8b5a2b',                                                                  # classical, identity-trained
    'geometric': '#7a7a7a', 'lbph': '#7a7a7a',                                                # classical, untrained
}
GROUP_LABEL = {
    'facenet': 'deep, trained for identity', 'arcface': 'deep, trained for identity',
    'cosface': 'deep, trained for identity', 'adaface': 'deep, trained for identity',
    'fairface': 'deep, trained not for identity', 'clip': 'deep, trained not for identity',
    'dinov2': 'deep, trained not for identity', 'blendshapes': 'deep, trained not for identity',
    'fisherface': 'classical, trained for identity',
    'geometric': 'classical, hand-crafted (untrained)', 'lbph': 'classical, hand-crafted (untrained)',
}
DISPLAY_NAME = {'clip': 'CLIP', 'dinov2': 'DINOv2', 'lbph': 'LBPH'}

# Canonical ethnicity groups, mapping each dataset's own (differently-granular) labels
# onto a shared name where the underlying group is genuinely the same population, so
# they can be pooled across datasets. Only (dataset, ethnicity) pairs that actually
# clear the >=20-image floor in tables/bias_ethnicity.csv appear here -- e.g. London's
# east_asian/west_asian/black groups never qualify (n=9/10/13), so London only
# contributes to White/Caucasian. Hispanic/Indian have no cross-dataset counterpart at
# all (MEBeauty-only) and are still shown, flagged as such.
ETHNICITY_CANONICAL = {
    ('scut', 'caucasian'): 'White/Caucasian',
    ('mebeauty', 'caucasian'): 'White/Caucasian',
    ('london', 'white'): 'White/Caucasian',
    ('scut', 'asian'): 'Asian',
    ('mebeauty', 'asian'): 'Asian',
    ('mebeauty', 'black'): 'Black',
    ('mebeauty', 'hispanic'): 'Hispanic*',
    ('mebeauty', 'indian'): 'Indian*',
}
ETHNICITY_SINGLE_DATASET = {'Hispanic*', 'Indian*'}  # MEBeauty-only, footnoted in the figure


def display(fam):
    return DISPLAY_NAME.get(fam, fam.capitalize())


def legend_handles():
    seen, handles = set(), []
    for fam, lab in GROUP_LABEL.items():
        if lab not in seen:
            seen.add(lab)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[fam], label=lab))
    return handles


def fig_accuracy_by_family():
    acc = defaultdict(dict)  # family -> dataset -> pearson
    with open(os.path.join(TAB, 'within.csv')) as f:
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
    ax.set_xticklabels([display(f) for f in families], rotation=20, ha='right')
    ax.set_ylabel('Attractiveness prediction accuracy\n(Pearson r, ridge probe, 10-fold CV)')
    ax.set_title('Within-dataset accuracy by embedding family and dataset')
    ax.legend(title='Dataset', frameon=False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RES, 'accuracy_by_family.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


def fig_invariance_and_accuracy_bars():
    rows = list(csv.DictReader(open(os.path.join(TAB, 'invariance.csv'))))
    rows.sort(key=lambda r: -float(r['dprime_pose']))
    fams = [r['family'] for r in rows]
    dp = [float(r['dprime_pose']) for r in rows]
    acc = [float(r['mean_within_r']) for r in rows]
    colors = [GROUP_COLOR[f] for f in fams]
    labels = [display(f) for f in fams]
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


def fig_bias_gap_bars():
    """Two-panel: mean ethnicity gap (top) and mean gender gap (bottom) by family.

    'Gap' = max subgroup Pearson r - min subgroup Pearson r for that family/dataset/
    attribute (tables/bias_summary.csv); averaged here across whichever datasets had
    >=2 qualifying subgroups for that attribute (ethnicity: scut+mebeauty only --
    London's non-white ethnicity groups are too small to qualify; gender: all 3).
    """
    rows = list(csv.DictReader(open(os.path.join(TAB, 'bias_summary.csv'))))
    eth_gap, gender_gap = defaultdict(list), defaultdict(list)
    for r in rows:
        if r['attribute'] == 'ethnicity':
            eth_gap[r['family']].append(float(r['gap']))
        elif r['attribute'] == 'gender':
            gender_gap[r['family']].append(float(r['gap']))
    eth_gap = {f: np.mean(v) for f, v in eth_gap.items()}
    gender_gap = {f: np.mean(v) for f, v in gender_gap.items()}

    fams = sorted(eth_gap, key=lambda f: -(eth_gap[f] + gender_gap[f]))
    colors = [GROUP_COLOR[f] for f in fams]
    labels = [display(f) for f in fams]
    x = np.arange(len(fams))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.bar(x, [eth_gap[f] for f in fams], color=colors)
    ax1.set_ylabel('mean ethnicity gap\n(max - min subgroup r)')
    ax1.set_title("Subgroup accuracy gap by family -- larger gap = less consistent\n"
                  "across demographic groups (sorted by combined ethnicity + gender gap)",
                  fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    ax2.bar(x, [gender_gap[f] for f in fams], color=colors)
    ax2.set_ylabel('mean gender gap\n(max - min subgroup r)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(handles=legend_handles(), frameon=False, loc='upper right', fontsize=9)

    fig.tight_layout()
    out = os.path.join(RES, 'bias_gap_by_family.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


def fig_bias_ethnicity_heatmap():
    """Family x ethnicity-group accuracy grid, MEBeauty only (the one dataset with all
    5 ethnicity groups at usable n). Columns ordered by pooled mean r descending, rows
    ordered by mean within-dataset accuracy descending (same order as accuracy_by_family.png).
    """
    rows = [r for r in csv.DictReader(open(os.path.join(TAB, 'bias_ethnicity.csv')))
            if r['dataset'] == 'mebeauty']
    by_fam = defaultdict(dict)
    for r in rows:
        by_fam[r['family']][r['ethnicity']] = float(r['pearson'])

    acc = defaultdict(dict)
    with open(os.path.join(TAB, 'within.csv')) as f:
        for r in csv.DictReader(f):
            if r['probe'] == 'ridge':
                acc[r['family']][r['dataset']] = float(r['pearson'])
    fams = sorted(by_fam, key=lambda f: -np.mean(list(acc[f].values())))

    eth_mean = defaultdict(list)
    for fam, d in by_fam.items():
        for eth, r in d.items():
            eth_mean[eth].append(r)
    eths = sorted(eth_mean, key=lambda e: -np.mean(eth_mean[e]))

    grid = np.array([[by_fam[f].get(e, np.nan) for e in eths] for f in fams])

    fig, ax = plt.subplots(figsize=(7, 7.5))
    im = ax.imshow(grid, cmap='Blues', vmin=0.2, vmax=1.0, aspect='auto')
    for i in range(len(fams)):
        for j in range(len(eths)):
            v = grid[i, j]
            color = 'white' if v > 0.66 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=color, fontsize=9)

    ax.set_xticks(np.arange(len(eths)))
    ax.set_xticklabels([e.capitalize() for e in eths], rotation=20, ha='right')
    ax.set_yticks(np.arange(len(fams)))
    ax.set_yticklabels([display(f) for f in fams])
    ax.set_title("MEBeauty accuracy by ethnicity group and family\n"
                  "(Pearson r, ridge probe, 10-fold CV; columns sorted by mean r)", fontsize=12)
    fig.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    fig.tight_layout()
    out = os.path.join(RES, 'bias_ethnicity_heatmap.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


def fig_accuracy_aggregate_by_family():
    """Single aggregate accuracy score per family: mean within-dataset Pearson r
    across all 3 datasets (same 'mean' number reported in README.md's within-dataset
    table), one bar per family instead of accuracy_by_family.png's per-dataset bars.
    """
    acc = defaultdict(list)
    with open(os.path.join(TAB, 'within.csv')) as f:
        for r in csv.DictReader(f):
            if r['probe'] == 'ridge':
                acc[r['family']].append(float(r['pearson']))
    agg = {f: np.mean(v) for f, v in acc.items()}
    fams = sorted(agg, key=lambda f: -agg[f])
    colors = [GROUP_COLOR[f] for f in fams]
    labels = [display(f) for f in fams]
    x = np.arange(len(fams))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, [agg[f] for f in fams], color=colors)
    for i, f in enumerate(fams):
        ax.text(i, agg[f] + 0.015, f'{agg[f]:.2f}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Mean within-dataset accuracy\n(Pearson r, averaged across SCUT/MEBeauty/London)')
    ax.set_title('Aggregate accuracy by family (across all 3 datasets)')
    ax.legend(handles=legend_handles(), frameon=False, loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RES, 'accuracy_aggregate_by_family.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


def fig_bias_aggregate_by_ethnicity():
    """Single aggregate accuracy score per canonical ethnicity group, pooled across
    every (family, dataset) row that maps to it (see ETHNICITY_CANONICAL), weighted by
    each row's subgroup n. Since n only varies by dataset (all 11 families share the
    same subgroup image count within one dataset), this is equivalent to averaging
    unweighted across families within a dataset, then combining datasets weighted by
    how many images support that group there.
    """
    rows = list(csv.DictReader(open(os.path.join(TAB, 'bias_ethnicity.csv'))))
    weighted_sum, weight_total = defaultdict(float), defaultdict(float)
    for r in rows:
        canon = ETHNICITY_CANONICAL.get((r['dataset'], r['ethnicity']))
        if canon is None:
            continue
        n = float(r['n'])
        weighted_sum[canon] += float(r['pearson']) * n
        weight_total[canon] += n
    agg = {g: weighted_sum[g] / weight_total[g] for g in weighted_sum}
    groups = sorted(agg, key=lambda g: -agg[g])
    tick_labels = [g + ('\n(MEBeauty only)' if g in ETHNICITY_SINGLE_DATASET else '') for g in groups]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(np.arange(len(groups)), [agg[g] for g in groups], color='#4a7c59')
    for i, g in enumerate(groups):
        ax.text(i, agg[g] + 0.01, f'{agg[g]:.3f}', ha='center', fontsize=9)

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel('Aggregate accuracy\n(Pearson r, n-weighted mean across families & datasets)')
    ax.set_title("Aggregate accuracy by ethnicity group\n"
                 "(all 11 families, pooled across whichever datasets have that group)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RES, 'bias_aggregate_by_ethnicity.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    fig_accuracy_by_family()
    fig_invariance_and_accuracy_bars()
    fig_bias_gap_bars()
    fig_bias_ethnicity_heatmap()
    fig_accuracy_aggregate_by_family()
    fig_bias_aggregate_by_ethnicity()
