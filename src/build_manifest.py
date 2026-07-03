"""Build unified manifest.csv across SCUT-FBP5500, MEBeauty, London Set.

Schema: dataset, image_id, identity, path, score, score_scale, gender, ethnicity, view, rated
- path is relative to repo root
- score is the mean human attractiveness rating in the dataset's native scale
- view: frontal for SCUT/MEBeauty; one of 10 view names for London
- rated: 1 if this exact image carries the attractiveness rating
  (London: only neutral_front images were rated; other views inherit identity but rated=0)
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []

# ---------------- SCUT-FBP5500 ----------------
scut_labels = os.path.join(ROOT, 'data/scut/SCUT-FBP5500_v2/train_test_files/All_labels.txt')
GENDER = {'F': 'female', 'M': 'male'}
ETH = {'A': 'asian', 'C': 'caucasian'}
with open(scut_labels) as f:
    for line in f:
        fn, score = line.split()
        rows.append(dict(
            dataset='scut', image_id=fn, identity=f'scut_{fn}',
            path=f'data/scut/SCUT-FBP5500_v2/Images/{fn}',
            score=float(score), score_scale='1-5',
            gender=GENDER[fn[1]], ethnicity=ETH[fn[0]],
            view='frontal', rated=1))

# ---------------- MEBeauty ----------------
for split in ('train', 'test'):
    p = os.path.join(ROOT, f'data/MEBeauty/scores/{split}_universal_scores.csv')
    with open(p) as f:
        for r in csv.DictReader(f):
            rel = r['image'].replace('images/', 'original_images/')
            parts = rel.split('/')  # original_images/<gender>/<ethnicity>/<file>
            rows.append(dict(
                dataset='mebeauty', image_id=parts[-1], identity=f'meb_{parts[-1]}',
                path=f'data/MEBeauty/{rel}',
                score=float(r['score']), score_scale='1-10',
                gender=parts[1], ethnicity=parts[2],
                view='frontal', rated=1))

# ---------------- London Set ----------------
info = {}
with open(os.path.join(ROOT, 'data/london/london_faces_info.csv')) as f:
    for r in csv.DictReader(f):
        info[r['face_id']] = r

means = {}
with open(os.path.join(ROOT, 'data/london/london_faces_ratings.csv')) as f:
    reader = csv.reader(f)
    hdr = next(reader)
    face_cols = hdr[3:]
    sums = defaultdict(float)
    counts = defaultdict(int)
    for row in reader:
        for col, val in zip(face_cols, row[3:]):
            if val.strip():
                sums[col] += float(val)
                counts[col] += 1
    for col in face_cols:
        means[col.lstrip('X')] = sums[col] / counts[col]

VIEWS = ['neutral_front', 'smiling_front',
         'neutral_left_3quarter', 'neutral_right_3quarter',
         'neutral_left_profile', 'neutral_right_profile',
         'smiling_left_3quarter', 'smiling_right_3quarter',
         'smiling_left_profile', 'smiling_right_profile']
for view in VIEWS:
    d = os.path.join(ROOT, 'data/london', view)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.jpg'):
            continue
        fid = fn.split('_')[0]  # e.g. 001_03.jpg -> 001
        if fid not in info:
            continue
        rows.append(dict(
            dataset='london', image_id=f'{view}/{fn}', identity=f'lon_{fid}',
            path=f'data/london/{view}/{fn}',
            score=means[fid], score_scale='1-7',
            gender=info[fid]['face_gender'], ethnicity=info[fid]['face_eth'],
            view=view, rated=1 if view == 'neutral_front' else 0))

out = os.path.join(ROOT, 'manifest.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# sanity report
missing = [r['path'] for r in rows if not os.path.exists(os.path.join(ROOT, r['path']))]
by_ds = defaultdict(int)
for r in rows:
    by_ds[r['dataset']] += 1
print('rows per dataset:', dict(by_ds))
print('rated rows:', sum(r['rated'] for r in rows))
print('missing files:', len(missing), missing[:5])
