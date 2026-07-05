"""Pose-invariance analysis (the novel contribution).

Data: London Set = 102 identities x 10 views (5 poses x 2 expressions), all pushed
through the identical preprocessing + embedding pipeline.

Invariance measure: d' pose-robust identity discriminability (on L2-normalized
embeddings). For each identity, compare the frontal embedding to each other neutral-
pose view (same-identity, cross-pose similarity) against frontal-vs-frontal similarity
between different identities (different-identity, same-pose similarity):

    d' = (mean_same - mean_diff) / pooled_std

Higher d' = the embedding more confidently keeps identity separable across pose change
-- i.e. more pose-invariant in the sense that matters for face-recognition robustness.
This is a standard signal-detection framing (matches how pose robustness is evaluated
in face-verification benchmarks), unlike a naive variance decomposition which fails to
discriminate among trained models and is not used here.

Then: correlate per-family d' against per-family attractiveness-prediction accuracy
(mean within-dataset and mean cross-dataset ridge Pearson r from results/within.csv,
results/cross.csv). The headline correlation is computed over the TRAINED embeddings
(see DEEP_FAMILIES): geometric and lbph are hand-crafted/classical with no learning at
all, never optimized to trade off pose-invariance for anything, so folding them into a
trend about *trained* invariance vs accuracy dilutes rather than clarifies. Their d'
and accuracy are still reported in the table, and the all-family correlation is printed
for transparency.

Usage: python src/invariance.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, 'embeddings')
RES = os.path.join(ROOT, 'results')
PRES = os.path.join(ROOT, 'presentables')
os.makedirs(RES, exist_ok=True)

FAMILIES = ['facenet', 'arcface', 'cosface', 'adaface', 'fairface', 'geometric', 'clip',
            'dinov2', 'blendshapes', 'lbph', 'fisherface']
# "trained" = underwent some learning process (regardless of objective) -- matches how
# fairface/clip were already included alongside the identity models. blendshapes is a
# trained regressor (mediapipe), dinov2 is trained self-supervised, fisherface is a
# classical LDA explicitly fit on identity labels (London). geometric and lbph are the
# only two families with NO learning at all (hand-crafted ratios / classical texture
# histogram) and stay excluded from the headline correlation, same as before.
DEEP_FAMILIES = {'facenet', 'arcface', 'cosface', 'adaface', 'fairface', 'clip',
                  'dinov2', 'blendshapes', 'fisherface'}
HAND_CRAFTED_FAMILIES = {'geometric', 'lbph'}

# Four-way taxonomy for the by-type breakdown: crossing (deep vs. classical) with
# (trained explicitly for identity vs. not). This is finer than DEEP_FAMILIES/
# HAND_CRAFTED_FAMILIES above -- e.g. fisherface is classical like geometric/lbph, but
# (unlike them) explicitly trained/fit to maximize identity discriminability, so it's
# the classical analog of the deep identity-margin models, not of geometric/lbph.
CATEGORIES = {
    'deep, trained for identity': ['facenet', 'arcface', 'cosface', 'adaface'],
    'deep, trained not for identity': ['fairface', 'clip', 'dinov2', 'blendshapes'],
    'classical, trained for identity': ['fisherface'],
    'classical, hand-crafted (untrained)': ['geometric', 'lbph'],
}
NEUTRAL_VIEWS = ['neutral_front', 'neutral_left_3quarter', 'neutral_right_3quarter',
                 'neutral_left_profile', 'neutral_right_profile']


def load_london(family):
    """Return {identity: {view: embedding}} with L2-normalized embeddings."""
    npz = np.load(os.path.join(EMB, f'{family}__london.npz'), allow_pickle=True)
    ids, X = list(npz['ids']), npz['X']
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    grid = defaultdict(dict)
    for iid, x in zip(ids, X):
        view, fn = str(iid).split('/')
        identity = fn.split('_')[0]
        grid[identity][view] = x
    return grid


def dprime(grid, views):
    """d' separating same-identity cross-pose similarities from different-identity
    frontal similarities. Returns (d', n_identities_used)."""
    same, diff = [], []
    idents = [i for i in grid if 'neutral_front' in grid[i]]
    for ident in idents:
        f = grid[ident]['neutral_front']
        for v in views:
            if v != 'neutral_front' and v in grid[ident]:
                same.append(float(f @ grid[ident][v]))
    fronts = {i: grid[i]['neutral_front'] for i in idents}
    keys = sorted(fronts)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            diff.append(float(fronts[keys[i]] @ fronts[keys[j]]))
    same, diff = np.array(same), np.array(diff)
    pooled = np.sqrt((same.var() + diff.var()) / 2)
    return float((same.mean() - diff.mean()) / (pooled + 1e-10)), len(idents)


def report_correlation(rows_subset, label):
    x = np.array([r['dprime_pose'] for r in rows_subset])
    yw = np.array([r['mean_within_r'] for r in rows_subset])
    pr, pp = pearsonr(x, yw)
    sr, sp = spearmanr(x, yw)
    print(f"\n[{label}] d' vs within-dataset accuracy: "
          f'Pearson r={pr:.3f} (p={pp:.3f}), Spearman rho={sr:.3f} (p={sp:.3f})')
    has_cross = all(r['mean_cross_r'] != '' for r in rows_subset)
    if has_cross:
        yc = np.array([r['mean_cross_r'] for r in rows_subset])
        prc, ppc = pearsonr(x, yc)
        src, spc = spearmanr(x, yc)
        print(f"[{label}] d' vs cross-dataset accuracy:  "
              f'Pearson r={prc:.3f} (p={ppc:.3f}), Spearman rho={src:.3f} (p={spc:.3f})')
    return x, yw


def report_by_category(rows):
    """Analyze each of the 4 (deep/classical x identity/not-identity) categories
    separately, then print a compact cross-category comparison table."""
    by_family = {r['family']: r for r in rows}
    summary = []
    print('\n' + '=' * 70)
    print('BY CATEGORY (analyzed separately)')
    print('=' * 70)
    for cat, fams in CATEGORIES.items():
        cat_rows = [by_family[f] for f in fams if f in by_family]
        if not cat_rows:
            continue
        print(f'\n[{cat}] ({len(cat_rows)} families)')
        for r in cat_rows:
            cross_str = f"{r['mean_cross_r']:.3f}" if r['mean_cross_r'] != '' else 'n/a'
            print(f"  {r['family']:12s} dprime={r['dprime_pose']:6.2f}  "
                  f"within_r={r['mean_within_r']:.3f}  cross_r={cross_str}")
        mean_dp = float(np.mean([r['dprime_pose'] for r in cat_rows]))
        mean_w = float(np.mean([r['mean_within_r'] for r in cat_rows]))
        cross_vals = [r['mean_cross_r'] for r in cat_rows if r['mean_cross_r'] != '']
        mean_c = float(np.mean(cross_vals)) if cross_vals else None
        summary.append(dict(category=cat, n=len(cat_rows), mean_dprime=round(mean_dp, 3),
                            mean_within_r=round(mean_w, 4),
                            mean_cross_r=round(mean_c, 4) if mean_c is not None else ''))
        if len(cat_rows) >= 3:
            x = np.array([r['dprime_pose'] for r in cat_rows])
            yw = np.array([r['mean_within_r'] for r in cat_rows])
            pr, pp = pearsonr(x, yw)
            print(f"  within-category d' vs within-r: Pearson r={pr:.3f} (p={pp:.3f})")
        else:
            print('  (too few families in this category for its own correlation)')

    print('\n' + '-' * 70)
    print('CROSS-CATEGORY COMPARISON')
    print('-' * 70)
    header = f"{'category':38s} {'n':>3s} {'mean d\'':>9s} {'mean within-r':>14s} {'mean cross-r':>13s}"
    print(header)
    for s in summary:
        cross_str = f"{s['mean_cross_r']:.3f}" if s['mean_cross_r'] != '' else 'n/a'
        print(f"{s['category']:38s} {s['n']:3d} {s['mean_dprime']:9.2f} "
              f"{s['mean_within_r']:14.3f} {cross_str:>13s}")

    with open(os.path.join(RES, 'category_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print('\nwrote results/category_summary.csv')


def main():
    within = defaultdict(list)
    with open(os.path.join(RES, 'within.csv')) as f:
        for r in csv.DictReader(f):
            if r['probe'] == 'ridge':
                within[r['family']].append(float(r['pearson']))
    cross = defaultdict(list)
    cpath = os.path.join(RES, 'cross.csv')
    if os.path.exists(cpath):
        with open(cpath) as f:
            for r in csv.DictReader(f):
                if r['probe'] == 'ridge':
                    cross[r['family']].append(float(r['pearson']))

    rows = []
    for fam in FAMILIES:
        path = os.path.join(EMB, f'{fam}__london.npz')
        if not os.path.exists(path) or fam not in within:
            continue
        grid = load_london(fam)
        dp, n_used = dprime(grid, NEUTRAL_VIEWS)
        mean_within = float(np.mean(within[fam]))
        mean_cross = float(np.mean(cross[fam])) if cross[fam] else ''
        rows.append(dict(family=fam, dprime_pose=round(dp, 3), n_identities=n_used,
                         mean_within_r=round(mean_within, 4),
                         mean_cross_r=round(mean_cross, 4) if mean_cross != '' else ''))
        cross_str = f'{mean_cross:.3f}' if mean_cross != '' else 'n/a'
        print(f'{fam:10s} dprime={dp:6.2f}  mean_within_r={mean_within:.3f}  '
              f'mean_cross_r={cross_str}', flush=True)

    with open(os.path.join(RES, 'invariance.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    deep_rows = [r for r in rows if r['family'] in DEEP_FAMILIES]
    report_correlation(rows, f'all {len(rows)} families')
    x, yw = report_correlation(deep_rows, f'{len(deep_rows)} trained families, excl. hand-crafted')
    report_by_category(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pr, pp = pearsonr(x, yw)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, yw, s=70, color='#c1440e')
    for r in deep_rows:
        ax.annotate(r['family'], (r['dprime_pose'], r['mean_within_r']),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)
    m, b = np.polyfit(x, yw, 1)
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, m * xs + b, '--', color='gray', alpha=0.7)
    ax.set_xlabel("Pose-robust identity discriminability  d'  (London Set)")
    ax.set_ylabel('Attractiveness prediction accuracy\n(mean within-dataset Pearson r, ridge)')
    ax.set_title(f"Invariance vs. attractiveness predictability ({len(deep_rows)} trained families)\n"
                 f"Pearson r={pr:.2f} (p={pp:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PRES, 'dprime_vs_accuracy.png'), dpi=150)
    print('\nwrote results/invariance.csv and presentables/dprime_vs_accuracy.png')


if __name__ == '__main__':
    main()
