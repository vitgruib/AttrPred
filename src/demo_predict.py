"""Given one raw photo, run the same detect+align pipeline as preprocess.py, extract
all 11 embedding-family features for that single photo (same models/preprocessing as
extract.py, just batch-of-one), fit each family's ridge probe on the full SCUT-rated
set (same recipe as evaluate.py's make_ridge(), fit on all of SCUT rather than
cross-validated since this is a "deployed probe" prediction, not an accuracy estimate),
and predict the attractiveness score. Also writes the 224x224 display crop used by the
demo_*_figure.py scripts.

This intentionally duplicates extract.py's per-family model-loading code in single-image
form rather than importing/refactoring it, so the validated dataset pipeline that
produced the published results can't be perturbed by this portfolio-illustration script.

Usage: python src/demo_predict.py <raw_photo.jpg> --display-out data/demo/demo_x.jpg --csv-out tables/demo_x.csv
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, 'embeddings')
sys.path.insert(0, os.path.join(ROOT, 'src'))

FAMILIES = ['facenet', 'arcface', 'cosface', 'adaface', 'fairface', 'geometric', 'clip',
            'dinov2', 'blendshapes', 'lbph', 'fisherface']


def detect_align_crop(app, img):
    """Mirrors preprocess.py: returns (aligned112 BGR, crop224 BGR) for the largest face."""
    from insightface.utils import face_align

    def detect(im):
        faces = app.get(im)
        if not faces:
            p = int(max(im.shape[:2]) * 0.4)
            im2 = cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_REFLECT)
            faces = app.get(im2)
            if faces:
                return faces, im2
        return faces, im

    faces, img = detect(img)
    if not faces:
        raise RuntimeError('no face detected')
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    aligned = face_align.norm_crop(img, face.kps, image_size=112)

    x1, y1, x2, y2 = face.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 1.3 / 2
    H, W = img.shape[:2]
    xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
    xb, yb = int(min(W, cx + half)), int(min(H, cy + half))
    crop224 = cv2.resize(img[ya:yb, xa:xb], (224, 224))

    return aligned, crop224, face, img


# ---------------- single-image embedders (mirrors extract.py, batch of one) ----------------

def embed_facenet(aligned_bgr, dev):
    import torch
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval().to(dev)
    img = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160)).astype(np.float32)
    img = (img - 127.5) / 128.0
    t = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(dev)
    with torch.no_grad():
        return model(t).cpu().numpy()[0]


def _insightface_rec_single(aligned_bgr, pack, model_file):
    from insightface import model_zoo
    from insightface.utils.storage import ensure_available
    pack_dir = ensure_available('models', pack, root=os.path.expanduser('~/.insightface'))
    cand = os.path.join(pack_dir, model_file)
    if not os.path.exists(cand):
        cand = os.path.join(pack_dir, pack, model_file)
    rec = model_zoo.get_model(cand, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    rec.prepare(ctx_id=0)
    return rec.get_feat(aligned_bgr).flatten()


def embed_arcface(aligned_bgr, dev):
    return _insightface_rec_single(aligned_bgr, 'buffalo_l', 'w600k_r50.onnx')


def embed_cosface(aligned_bgr, dev):
    return _insightface_rec_single(aligned_bgr, 'antelopev2', 'glintr100.onnx')


def embed_adaface(aligned_bgr, dev):
    import torch
    from extract import _load_adaface
    model = _load_adaface(dev)
    img = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)[None])).to(dev)
    with torch.no_grad():
        out = model(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.cpu().numpy()[0]


def embed_fairface(crop224_bgr, dev):
    import torch
    import torchvision
    ckpt = os.path.join(ROOT, 'models/res34_fair_align_multi_7_20190809.pt')
    model = torchvision.models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.fc = torch.nn.Identity()
    model = model.eval().to(dev)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = cv2.cvtColor(crop224_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - mean) / std
    t = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(dev)
    with torch.no_grad():
        return model(t).cpu().numpy()[0]


def embed_clip(crop224_bgr, dev):
    import torch
    import open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.eval().to(dev)
    pil = Image.fromarray(cv2.cvtColor(crop224_bgr, cv2.COLOR_BGR2RGB))
    t = preprocess(pil).unsqueeze(0).to(dev)
    with torch.no_grad():
        return model.encode_image(t).cpu().numpy()[0]


def embed_geometric(face):
    from extract import _kps_features
    feats = _kps_features(np.asarray(face.kps, dtype=np.float32), face.bbox)
    lms = getattr(face, 'landmark_2d_106', None)
    if lms is not None:
        lms = np.asarray(lms, dtype=np.float32)
        kps = np.asarray(face.kps, dtype=np.float32)
        s = np.linalg.norm(kps[0] - kps[1]) + 1e-8
        cloud = ((lms - lms.mean(0)) / s).flatten()
    else:
        cloud = np.zeros(0, dtype=np.float32)
    return np.concatenate([np.array(feats, dtype=np.float32), cloud])


def embed_dinov2(crop224_bgr, dev):
    import torch
    import timm
    from PIL import Image
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
    model = model.eval().to(dev)
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    pil = Image.fromarray(cv2.cvtColor(crop224_bgr, cv2.COLOR_BGR2RGB))
    t = transform(pil).unsqueeze(0).to(dev)
    with torch.no_grad():
        return model(t).cpu().numpy()[0]


def embed_blendshapes(crop224_bgr):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model_path = os.path.join(ROOT, 'models', 'face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    img = cv2.cvtColor(crop224_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    res = landmarker.detect(mp_img)
    if not res.face_blendshapes:
        raise RuntimeError('blendshapes: no face detected')
    return np.array([c.score for c in res.face_blendshapes[0]], dtype=np.float32)


def embed_lbph(aligned_bgr):
    # LBPH histograms are computed per-image independently of the rest of the training
    # set, so training a fresh recognizer on just this one image yields the same
    # descriptor extract.py would have produced training on the whole dataset.
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([gray], np.array([0]))
    return recognizer.getHistograms()[0].flatten()


def embed_fisherface(aligned_bgr):
    from extract import _fit_fisherface
    mean, eigvecs = _fit_fisherface()
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().astype(np.float64)
    return (flat - mean) @ eigvecs


def embed_all(raw_bgr, app, dev):
    aligned, crop224, face, _ = detect_align_crop(app, raw_bgr)
    return crop224, {
        'facenet': embed_facenet(aligned, dev),
        'arcface': embed_arcface(aligned, dev),
        'cosface': embed_cosface(aligned, dev),
        'adaface': embed_adaface(aligned, dev),
        'fairface': embed_fairface(crop224, dev),
        'geometric': embed_geometric(face),
        'clip': embed_clip(crop224, dev),
        'dinov2': embed_dinov2(crop224, dev),
        'blendshapes': embed_blendshapes(crop224),
        'lbph': embed_lbph(aligned),
        'fisherface': embed_fisherface(aligned),
    }


# ---------------- ridge probe, fit on all of SCUT (same recipe as evaluate.py) ----------------

def predict_scores(embeddings_by_family):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    with open(os.path.join(ROOT, 'manifest.csv')) as f:
        man = {r['image_id']: r for r in csv.DictReader(f) if r['dataset'] == 'scut'}

    alphas = np.logspace(-2, 5, 15)
    scores = {}
    for fam in FAMILIES:
        npz = np.load(os.path.join(EMB, f'{fam}__scut.npz'), allow_pickle=True)
        ids, X = list(npz['ids']), npz['X']
        keep, y = [], []
        for i, iid in enumerate(ids):
            r = man.get(str(iid))
            if r and r['rated'] == '1':
                keep.append(i)
                y.append(float(r['score']))
        Xk = np.nan_to_num(X[keep], nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(y)

        probe = make_pipeline(StandardScaler(),
                               GridSearchCV(Ridge(), {'alpha': alphas}, cv=3, scoring='neg_mean_squared_error'))
        probe.fit(Xk, y)

        x_new = np.nan_to_num(embeddings_by_family[fam], nan=0.0, posinf=0.0, neginf=0.0)[None]
        scores[fam] = float(probe.predict(x_new)[0])
        print(f'{fam:10s} predicted={scores[fam]:.3f}', flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo')
    ap.add_argument('--display-out', required=True)
    ap.add_argument('--csv-out', required=True)
    args = ap.parse_args()

    import torch
    from insightface.app import FaceAnalysis
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    raw = cv2.imread(args.photo)
    if raw is None:
        raise SystemExit(f'could not read {args.photo}')

    crop224, embeddings = embed_all(raw, app, dev)
    cv2.imwrite(args.display_out, crop224)
    print('wrote', args.display_out)

    scores = predict_scores(embeddings)
    with open(args.csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family', 'predicted_score'])
        for fam in FAMILIES:
            w.writerow([fam, scores[fam]])
    print('wrote', args.csv_out)


if __name__ == '__main__':
    main()
