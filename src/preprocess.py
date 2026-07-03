"""Detect + align faces for every image in manifest.csv.

One detector for all datasets (RetinaFace via insightface buffalo_l), then:
  - aligned 112x112 (ArcFace 5-point template)  -> data/aligned112/<dataset>/<safe_id>.jpg
    (margin models consume this directly; FaceNet consumes it resized to 160)
  - loose 224x224 crop (1.3x expanded bbox)     -> data/crops224/<dataset>/<safe_id>.jpg
    (CLIP / FairFace / MediaPipe consume this)
  - preprocess_meta.csv: image_id, dataset, det_score, yaw, pitch, roll, ok

Largest face wins when multiple are detected. Failures logged with ok=0.
"""
import csv
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def safe_id(image_id: str) -> str:
    return image_id.replace('/', '__')


def main():
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    with open(os.path.join(ROOT, 'manifest.csv')) as f:
        manifest = list(csv.DictReader(f))

    meta_rows = []
    n_fail = 0
    for i, row in enumerate(manifest):
        ds, iid = row['dataset'], row['image_id']
        sid = safe_id(iid)
        out112 = os.path.join(ROOT, 'data/aligned112', ds, sid)
        out224 = os.path.join(ROOT, 'data/crops224', ds, sid)
        os.makedirs(os.path.dirname(out112), exist_ok=True)
        os.makedirs(os.path.dirname(out224), exist_ok=True)

        if os.path.exists(out112) and os.path.exists(out224):
            # already done (resume support); meta row recovered below only for new runs
            continue

        img = cv2.imread(os.path.join(ROOT, row['path']))
        if img is None:
            meta_rows.append(dict(dataset=ds, image_id=iid, det_score=0, yaw='', pitch='', roll='', ok=0, reason='read_fail'))
            n_fail += 1
            continue

        faces = app.get(img)
        if not faces:
            # retry with reflect-padded border: RetinaFace misses frame-filling
            # faces that touch the image edges (common in tightly-cropped SCUT/MEBeauty)
            h, w = img.shape[:2]
            p = int(max(h, w) * 0.4)
            img2 = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT)
            faces = app.get(img2)
            if faces:
                img = img2
        if not faces:
            # last resort: upscale small images and pad
            h, w = img.shape[:2]
            if max(h, w) < 400:
                img2 = cv2.resize(img, (w * 2, h * 2))
                p = int(max(img2.shape[:2]) * 0.4)
                img2 = cv2.copyMakeBorder(img2, p, p, p, p, cv2.BORDER_REFLECT)
                faces = app.get(img2)
                if faces:
                    img = img2
        if not faces:
            meta_rows.append(dict(dataset=ds, image_id=iid, det_score=0, yaw='', pitch='', roll='', ok=0, reason='no_face'))
            n_fail += 1
            continue

        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        # 112x112 ArcFace alignment from 5-point landmarks
        aligned = face_align.norm_crop(img, face.kps, image_size=112)
        cv2.imwrite(out112, aligned)

        # loose square crop, 1.3x bbox expansion
        x1, y1, x2, y2 = face.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * 1.3 / 2
        H, W = img.shape[:2]
        xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
        xb, yb = int(min(W, cx + half)), int(min(H, cy + half))
        crop = cv2.resize(img[ya:yb, xa:xb], (224, 224))
        cv2.imwrite(out224, crop)

        pose = getattr(face, 'pose', None)
        pitch, yaw, roll = (pose if pose is not None else ('', '', ''))
        meta_rows.append(dict(dataset=ds, image_id=iid, det_score=float(face.det_score),
                              yaw=yaw, pitch=pitch, roll=roll, ok=1, reason=''))

        if (i + 1) % 500 == 0:
            print(f'{i + 1}/{len(manifest)} processed, {n_fail} failures', flush=True)

    out_meta = os.path.join(ROOT, 'preprocess_meta.csv')
    write_header = not os.path.exists(out_meta)
    with open(out_meta, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image_id', 'det_score', 'yaw', 'pitch', 'roll', 'ok', 'reason'])
        if write_header:
            w.writeheader()
        w.writerows(meta_rows)
    print(f'done. {len(meta_rows)} new, {n_fail} failures -> preprocess_meta.csv', flush=True)


if __name__ == '__main__':
    sys.exit(main())
