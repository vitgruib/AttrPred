"""Train identical lightweight probes on each embedding family; evaluate within- and cross-dataset.

Within-dataset: 5-fold CV, ridge regression (alpha tuned on inner CV) + small MLP (secondary).
Cross-dataset:  train on all of A, test on all of B (scores z-normalized per dataset;
                Pearson/Spearman are scale-invariant, RMSE reported in z-units).

Only rated==1 rows are used (London: neutral_front only).

Outputs: results/within.csv, results/cross.csv
Usage: python src/evaluate.py [--mlp]
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, 'embeddings')
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

FAMILIES = ['facenet', 'arcface', 'cosface', 'adaface', 'fairface', 'geometric', 'clip']
DATASETS = ['scut', 'mebeauty', 'london']
SEED = 42
ALPHAS = np.logspace(-2, 5, 15)


def load_xy(family, dataset):
    """Return X, y, meta rows aligned; rated rows only."""
    npz = np.load(os.path.join(EMB, f'{family}__{dataset}.npz'), allow_pickle=True)
    ids = list(npz['ids'])
    X = npz['X']
    with open(os.path.join(ROOT, 'manifest.csv')) as f:
        man = {r['image_id']: r for r in csv.DictReader(f) if r['dataset'] == dataset}
    keep, y, meta = [], [], []
    for i, iid in enumerate(ids):
        r = man.get(str(iid))
        if r and r['rated'] == '1':
            keep.append(i)
            y.append(float(r['score']))
            meta.append(r)
    Xk = np.nan_to_num(X[keep], nan=0.0, posinf=0.0, neginf=0.0)
    return Xk, np.array(y), meta


def make_ridge():
    return make_pipeline(
        StandardScaler(),
        GridSearchCV(Ridge(), {'alpha': ALPHAS}, cv=3, scoring='neg_mean_squared_error'))


def make_mlp():
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True,
                     max_iter=500, random_state=SEED))


def within_dataset(probe_name, make_probe):
    rows = []
    for fam in FAMILIES:
        for ds in DATASETS:
            path = os.path.join(EMB, f'{fam}__{ds}.npz')
            if not os.path.exists(path):
                continue
            X, y, _ = load_xy(fam, ds)
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            preds = np.zeros_like(y)
            for tr, te in kf.split(X):
                m = make_probe()
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])
            r = pearsonr(y, preds)[0]
            rho = spearmanr(y, preds)[0]
            rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
            rows.append(dict(probe=probe_name, family=fam, dataset=ds, n=len(y),
                             pearson=round(r, 4), spearman=round(rho, 4), rmse=round(rmse, 4)))
            print(f'[within/{probe_name}] {fam:10s} {ds:9s} n={len(y):5d} r={r:.3f} rho={rho:.3f}', flush=True)
    return rows


def cross_dataset(probe_name, make_probe):
    rows = []
    for fam in FAMILIES:
        data = {}
        for ds in DATASETS:
            path = os.path.join(EMB, f'{fam}__{ds}.npz')
            if os.path.exists(path):
                X, y, _ = load_xy(fam, ds)
                data[ds] = (X, (y - y.mean()) / y.std())
        for a in data:
            for b in data:
                if a == b:
                    continue
                Xa, ya = data[a]
                Xb, yb = data[b]
                m = make_probe()
                m.fit(Xa, ya)
                p = m.predict(Xb)
                r = pearsonr(yb, p)[0]
                rho = spearmanr(yb, p)[0]
                rows.append(dict(probe=probe_name, family=fam, train=a, test=b,
                                 pearson=round(r, 4), spearman=round(rho, 4)))
                print(f'[cross/{probe_name}] {fam:10s} {a:9s}->{b:9s} r={r:.3f} rho={rho:.3f}', flush=True)
    return rows


def write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote', path)


if __name__ == '__main__':
    probes = [('ridge', make_ridge)]
    if '--mlp' in sys.argv:
        probes.append(('mlp', make_mlp))
    within_rows, cross_rows = [], []
    for name, mk in probes:
        within_rows += within_dataset(name, mk)
        cross_rows += cross_dataset(name, mk)
    write(os.path.join(RES, 'within.csv'), within_rows)
    write(os.path.join(RES, 'cross.csv'), cross_rows)
