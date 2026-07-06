"""Demographic bias breakdown: does attractiveness-prediction accuracy differ by subgroup?

For each (family, dataset), refit the ridge probe in the same 10-fold CV protocol as
evaluate.py, then compute Pearson r of predictions vs. true scores *within each
demographic subgroup*. This surfaces whether an embedding family predicts attractiveness
better for some groups than others -- a real confound risk, since every identity
backbone here was trained on an external face corpus (VGGFace2/WebFace600K/Glint360K/
WebFace4M/FairFace) with its own demographic skew, and that skew could leak into how well
the frozen embedding represents faces outside its training distribution.

Subgroups analyzed:
  - gender: all 3 datasets (native label in manifest.csv)
  - ethnicity: all 3 datasets (native label; category granularity differs a lot --
    SCUT is binary asian/caucasian, MEBeauty has 5 groups, London has 5 -- so gaps are
    only compared within a dataset, never across datasets, see README.md)
  - age: London only. SCUT and MEBeauty have no ground-truth age label at all; rather
    than estimate age with a separate pretrained model (a noisy, model-derived proxy
    that would itself be a new confound), age-subgroup analysis is restricted to the one
    dataset with real per-identity age (data/london/london_faces_info.csv, ages 18-54,
    n=100 of 102 identities have a recorded age). Buckets are tertiles computed at
    runtime (not fixed cutoffs) so each bucket clears the n>=20 inclusion floor below.

A subgroup is only reported if it has >= 20 rated images (same floor the original
bias analysis used) -- below that, a Pearson r is too noisy to be worth reporting.

Outputs:
  tables/bias_gender.csv     family, dataset, gender, n, pearson
  tables/bias_ethnicity.csv  family, dataset, ethnicity, n, pearson
  tables/bias_age.csv        family, dataset(london only), age_bucket, n, pearson
  tables/bias_summary.csv    family, dataset, attribute, n_groups, gap
    (gap = max subgroup pearson - min subgroup pearson; only rows with >=2 qualifying
    subgroups get a gap -- this is the single headline number per family/dataset/attribute)

Usage: python src/bias.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from evaluate import FAMILIES, DATASETS, SEED, TAB, load_xy, make_ridge

MIN_SUBGROUP_N = 20


def age_bucket_labels(meta):
    """Tertile age buckets computed from this dataset's own age values (London only)."""
    ages = [int(r['age']) for r in meta if r['age'].strip()]
    if len(ages) < 3 * MIN_SUBGROUP_N:
        return {}
    lo, hi = np.quantile(ages, [1 / 3, 2 / 3])
    labels = {}
    for i, r in enumerate(meta):
        if not r['age'].strip():
            continue
        a = int(r['age'])
        if a <= lo:
            labels[i] = f'<={int(lo)}'
        elif a <= hi:
            labels[i] = f'{int(lo) + 1}-{int(hi)}'
        else:
            labels[i] = f'>{int(hi)}'
    return labels


def subgroup_r(y, preds, group_of):
    """group_of: dict index -> group label (only indices present are analyzed)."""
    groups = defaultdict(list)
    for i, g in group_of.items():
        groups[g].append(i)
    out = {}
    for g, idx in groups.items():
        if len(idx) >= MIN_SUBGROUP_N:
            idx = np.array(idx)
            out[g] = (round(float(pearsonr(y[idx], preds[idx])[0]), 4), len(idx))
    return out


def main():
    gender_rows, eth_rows, age_rows, summary_rows = [], [], [], []
    for fam in FAMILIES:
        for ds in DATASETS:
            path = os.path.join(os.path.dirname(__file__), '..', 'embeddings', f'{fam}__{ds}.npz')
            if not os.path.exists(path):
                continue
            X, y, meta = load_xy(fam, ds)
            kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
            preds = np.zeros_like(y)
            for tr, te in kf.split(X):
                m = make_ridge()
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])

            gender_of = {i: r['gender'] for i, r in enumerate(meta)}
            eth_of = {i: r['ethnicity'] for i, r in enumerate(meta)}

            gender_res = subgroup_r(y, preds, gender_of)
            for g, (r, n) in gender_res.items():
                gender_rows.append(dict(family=fam, dataset=ds, gender=g, n=n, pearson=r))
            if len(gender_res) >= 2:
                vals = [r for r, _ in gender_res.values()]
                summary_rows.append(dict(family=fam, dataset=ds, attribute='gender',
                                          n_groups=len(gender_res), gap=round(max(vals) - min(vals), 4)))

            eth_res = subgroup_r(y, preds, eth_of)
            for g, (r, n) in eth_res.items():
                eth_rows.append(dict(family=fam, dataset=ds, ethnicity=g, n=n, pearson=r))
            if len(eth_res) >= 2:
                vals = [r for r, _ in eth_res.values()]
                summary_rows.append(dict(family=fam, dataset=ds, attribute='ethnicity',
                                          n_groups=len(eth_res), gap=round(max(vals) - min(vals), 4)))

            if ds == 'london':
                age_of = age_bucket_labels(meta)
                age_res = subgroup_r(y, preds, age_of)
                for g, (r, n) in age_res.items():
                    age_rows.append(dict(family=fam, dataset=ds, age_bucket=g, n=n, pearson=r))
                if len(age_res) >= 2:
                    vals = [r for r, _ in age_res.values()]
                    summary_rows.append(dict(family=fam, dataset=ds, attribute='age',
                                              n_groups=len(age_res), gap=round(max(vals) - min(vals), 4)))

            print(f'{fam:10s} {ds:9s} done', flush=True)

    def write(name, rows):
        out = os.path.join(TAB, f'{name}.csv')
        if not rows:
            print(f'skipped {out} (no qualifying subgroups)')
            return
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print('wrote', out)

    write('bias_gender', gender_rows)
    write('bias_ethnicity', eth_rows)
    write('bias_age', age_rows)
    write('bias_summary', summary_rows)


if __name__ == '__main__':
    main()
