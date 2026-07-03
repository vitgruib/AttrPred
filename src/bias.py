"""Bias/fairness breakdown: attractiveness-prediction accuracy by gender and ethnicity.

For each (family, dataset), refit the ridge probe in 5-fold CV (same protocol as evaluate.py),
then compute Pearson r of predictions vs true scores *within each demographic subgroup*.
This surfaces whether an embedding family predicts attractiveness better for some groups —
a confound risk because identity backbones (VGGFace2/MS1M/WebFace) have demographic skew.

Outputs: results/bias_gender.csv, results/bias_ethnicity.csv
Usage: python src/bias.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from evaluate import FAMILIES, DATASETS, SEED, load_xy, make_ridge


def subgroup_r(y, preds, meta, key):
    groups = defaultdict(list)
    for i, r in enumerate(meta):
        groups[r[key]].append(i)
    out = {}
    for g, idx in groups.items():
        if len(idx) >= 20:
            idx = np.array(idx)
            out[g] = (round(pearsonr(y[idx], preds[idx])[0], 4), len(idx))
    return out


def main():
    gender_rows, eth_rows = [], []
    for fam in FAMILIES:
        for ds in DATASETS:
            path = os.path.join(os.path.dirname(__file__), '..', 'embeddings', f'{fam}__{ds}.npz')
            if not os.path.exists(path):
                continue
            X, y, meta = load_xy(fam, ds)
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            preds = np.zeros_like(y)
            for tr, te in kf.split(X):
                m = make_ridge()
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])
            for g, (r, n) in subgroup_r(y, preds, meta, 'gender').items():
                gender_rows.append(dict(family=fam, dataset=ds, gender=g, n=n, pearson=r))
            for g, (r, n) in subgroup_r(y, preds, meta, 'ethnicity').items():
                eth_rows.append(dict(family=fam, dataset=ds, ethnicity=g, n=n, pearson=r))
            print(f'{fam:10s} {ds:9s} done', flush=True)

    RES = os.path.join(os.path.dirname(__file__), '..', 'results')
    for name, rows in [('bias_gender', gender_rows), ('bias_ethnicity', eth_rows)]:
        with open(os.path.join(RES, f'{name}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print('wrote results/bias_gender.csv, results/bias_ethnicity.csv')


if __name__ == '__main__':
    main()
