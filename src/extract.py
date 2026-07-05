"""Extract embeddings for all families on all datasets.

Usage: python src/extract.py <family> [dataset]
Families: facenet, arcface, cosface, adaface, fairface, geometric, clip,
          dinov2, blendshapes, lbph, fisherface
Output: embeddings/<family>__<dataset>.npz  {ids: [image_id], X: [n,d] float32}

Inputs produced by preprocess.py:
  data/aligned112/<dataset>/<safe_id>.jpg  (ArcFace 5pt template)  -> identity models, lbph, fisherface
  data/crops224/<dataset>/<safe_id>.jpg    (loose 1.3x crop)       -> clip/fairface/geometric/dinov2/blendshapes
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, 'embeddings')
os.makedirs(EMB, exist_ok=True)

DATASETS = ['scut', 'mebeauty', 'london']


def load_manifest(dataset):
    with open(os.path.join(ROOT, 'manifest.csv')) as f:
        return [r for r in csv.DictReader(f) if r['dataset'] == dataset]


def safe_id(image_id):
    return image_id.replace('/', '__')


def iter_images(dataset, subdir):
    """Yield (image_id, path) for images that were successfully preprocessed."""
    for r in load_manifest(dataset):
        p = os.path.join(ROOT, subdir, dataset, safe_id(r['image_id']))
        if os.path.exists(p):
            yield r['image_id'], p


def save(family, dataset, ids, X):
    out = os.path.join(EMB, f'{family}__{dataset}.npz')
    np.savez_compressed(out, ids=np.array(ids), X=np.asarray(X, dtype=np.float32))
    print(f'{out}: {len(ids)} x {X.shape[1]}', flush=True)


# ---------------- families ----------------

def run_facenet(dataset):
    import torch
    from facenet_pytorch import InceptionResnetV1
    import cv2
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InceptionResnetV1(pretrained='vggface2').eval().to(dev)
    ids, X, batch = [], [], []
    items = list(iter_images(dataset, 'data/aligned112'))
    with torch.no_grad():
        for i, (iid, p) in enumerate(items):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 160)).astype(np.float32)
            batch.append((img - 127.5) / 128.0)
            ids.append(iid)
            if len(batch) == 128 or i == len(items) - 1:
                t = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2)).to(dev)
                X.append(model(t).cpu().numpy())
                batch = []
    save('facenet', dataset, ids, np.concatenate(X))


def _insightface_rec(dataset, family, pack, model_file):
    """Shared path for arcface (buffalo_l) and cosface (antelopev2 glintr100)."""
    import cv2
    from insightface import model_zoo
    from insightface.utils.storage import ensure_available
    pack_dir = ensure_available('models', pack, root=os.path.expanduser('~/.insightface'))
    # some packs (antelopev2) extract into a nested subdir of the same name
    cand = os.path.join(pack_dir, model_file)
    if not os.path.exists(cand):
        cand = os.path.join(pack_dir, pack, model_file)
    rec = model_zoo.get_model(cand, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    rec.prepare(ctx_id=0)
    ids, X = [], []
    for iid, p in iter_images(dataset, 'data/aligned112'):
        img = cv2.imread(p)  # BGR, 112x112, already aligned
        X.append(rec.get_feat(img).flatten())
        ids.append(iid)
    save(family, dataset, ids, np.stack(X))


def run_arcface(dataset):
    _insightface_rec(dataset, 'arcface', 'buffalo_l', 'w600k_r50.onnx')


def run_cosface(dataset):
    _insightface_rec(dataset, 'cosface', 'antelopev2', 'glintr100.onnx')


def _load_adaface(dev):
    """Load cvlface AdaFace IR-101 (WebFace4M) via the repo's own models.get_model.
    The wrapper uses relative paths + `from models import get_model`, so we chdir
    into the snapshot dir and put it on sys.path, then restore cwd."""
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    repo = snapshot_download('minchul/cvlface_adaface_ir101_webface4m')
    cwd = os.getcwd()
    os.chdir(repo)
    sys.path.insert(0, repo)
    try:
        from models import get_model
        cfg = OmegaConf.create(dict(yaml.safe_load(open('pretrained_model/model.yaml'))))
        model = get_model(cfg)
        model.load_state_dict_from_path('pretrained_model/model.pt')
    finally:
        os.chdir(cwd)
    return model.eval().to(dev)


def run_adaface(dataset):
    import torch
    import cv2
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _load_adaface(dev)
    ids, X, batch = [], [], []
    items = list(iter_images(dataset, 'data/aligned112'))
    with torch.no_grad():
        for i, (iid, p) in enumerate(items):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
            batch.append((img / 255.0 - 0.5) / 0.5)  # RGB in [-1,1]
            ids.append(iid)
            if len(batch) == 64 or i == len(items) - 1:
                # .copy() -> contiguous; adaface's iresnet does a raw .view() that
                # rejects the non-contiguous array produced by transpose
                t = torch.from_numpy(np.ascontiguousarray(
                    np.stack(batch).transpose(0, 3, 1, 2))).to(dev)
                out = model(t)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                X.append(out.cpu().numpy())
                batch = []
    save('adaface', dataset, ids, np.concatenate(X))


def run_fairface(dataset):
    import torch
    import torchvision
    import cv2
    ckpt = os.path.join(ROOT, 'models/res34_fair_align_multi_7_20190809.pt')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torchvision.models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.fc = torch.nn.Identity()  # penultimate 512-d features
    model = model.eval().to(dev)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    ids, X, batch = [], [], []
    items = list(iter_images(dataset, 'data/crops224'))
    with torch.no_grad():
        for i, (iid, p) in enumerate(items):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append((img - mean) / std)
            ids.append(iid)
            if len(batch) == 128 or i == len(items) - 1:
                t = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2)).to(dev)
                X.append(model(t).cpu().numpy())
                batch = []
    save('fairface', dataset, ids, np.concatenate(X))


def run_clip(dataset):
    import torch
    import open_clip
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.eval().to(dev)
    ids, X, batch = [], [], []
    items = list(iter_images(dataset, 'data/crops224'))
    with torch.no_grad():
        for i, (iid, p) in enumerate(items):
            batch.append(preprocess(Image.open(p)))
            ids.append(iid)
            if len(batch) == 64 or i == len(items) - 1:
                t = torch.stack(batch).to(dev)
                X.append(model.encode_image(t).cpu().numpy())
                batch = []
    save('clip', dataset, ids, np.concatenate(X))


# -- geometric baseline: insightface 106-point 2D landmarks on the ORIGINAL image
#    (geometry must come from the unwarped image, not the aligned 112 crop). The
#    stripped mediapipe WSL wheel lacks the legacy FaceMesh solution, so we reuse
#    the buffalo_l detector already used for preprocessing -> same detector, robust.

def _kps_features(kps, bbox):
    """Ratio/symmetry features from the reliable 5-point kps + bbox.
    kps rows: 0 left-eye, 1 right-eye, 2 nose, 3 left-mouth, 4 right-mouth."""
    le, re, nose, lm, rm = kps
    inter_eye = np.linalg.norm(le - re) + 1e-8
    eye_mid = (le + re) / 2
    mouth_mid = (lm + rm) / 2
    mouth_w = np.linalg.norm(lm - rm)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    eye_to_mouth = np.linalg.norm(eye_mid - mouth_mid)
    return [
        face_w / (face_h + 1e-8),                          # face aspect
        inter_eye / (face_w + 1e-8),                       # eye spacing / width
        mouth_w / (inter_eye),                             # mouth/eye ratio
        eye_to_mouth / (inter_eye),                        # vertical thirds
        np.linalg.norm(nose - eye_mid) / eye_to_mouth,     # nose position
        np.linalg.norm(nose - mouth_mid) / eye_to_mouth,
        # horizontal symmetry: nose deviation from eye midline, normalized
        abs((nose[0] - eye_mid[0])) / inter_eye,
        abs(np.linalg.norm(le - nose) - np.linalg.norm(re - nose)) / inter_eye,
        abs(np.linalg.norm(lm - nose) - np.linalg.norm(rm - nose)) / inter_eye,
    ]


def run_geometric(dataset):
    import cv2
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    def detect(im):
        faces = app.get(im)
        if not faces:
            p = int(max(im.shape[:2]) * 0.4)
            im2 = cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_REFLECT)
            faces = app.get(im2)
        return faces

    ids, X = [], []
    n_fail = 0
    for r in load_manifest(dataset):
        img = cv2.imread(os.path.join(ROOT, r['path']))
        if img is None:
            n_fail += 1
            continue
        faces = detect(img)
        if not faces:
            n_fail += 1
            continue
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        feats = _kps_features(np.asarray(face.kps, dtype=np.float32), face.bbox)
        # normalized 106-point cloud (center on centroid, scale by interocular)
        lms = getattr(face, 'landmark_2d_106', None)
        if lms is not None:
            lms = np.asarray(lms, dtype=np.float32)
            kps = np.asarray(face.kps, dtype=np.float32)
            s = np.linalg.norm(kps[0] - kps[1]) + 1e-8
            cloud = ((lms - lms.mean(0)) / s).flatten()
        else:
            cloud = np.zeros(0, dtype=np.float32)
        vec = np.concatenate([np.array(feats, dtype=np.float32), cloud])
        X.append(vec)
        ids.append(r['image_id'])
    print(f'geometric {dataset}: {n_fail} detection failures, dim={len(X[0])}')
    save('geometric', dataset, ids, np.stack(X))


# -- dinov2: self-supervised ViT (no text/identity supervision at all -- an even
#    cleaner "never trained to discard appearance" contrast than CLIP)

def run_dinov2(dataset):
    import torch
    import timm
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
    model = model.eval().to(dev)
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    ids, X, batch = [], [], []
    items = list(iter_images(dataset, 'data/crops224'))
    with torch.no_grad():
        for i, (iid, p) in enumerate(items):
            batch.append(transform(Image.open(p).convert('RGB')))
            ids.append(iid)
            if len(batch) == 32 or i == len(items) - 1:
                t = torch.stack(batch).to(dev)
                X.append(model(t).cpu().numpy())
                batch = []
    save('dinov2', dataset, ids, np.concatenate(X))


# -- blendshapes: MediaPipe Face Landmarker's 52 ARKit blendshape scores. A trained
#    model, but trained to regress named local shape/expression coefficients, not to
#    discriminate identity -- same "not part of the identity-invariance tradeoff" story
#    as fairface/clip, just a richer hand-named feature space than Geometric's 9 ratios.

def run_blendshapes(dataset):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import cv2
    model_path = os.path.join(ROOT, 'models', 'face_landmarker.task')
    if not os.path.exists(model_path):
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
            'face_landmarker/float16/1/face_landmarker.task', model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    ids, X = [], []
    n_fail = 0
    for iid, p in iter_images(dataset, 'data/crops224'):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        res = landmarker.detect(mp_img)
        if not res.face_blendshapes:
            n_fail += 1
            continue
        X.append(np.array([c.score for c in res.face_blendshapes[0]], dtype=np.float32))
        ids.append(iid)
    print(f'blendshapes {dataset}: {n_fail} detection failures, dim={len(X[0])}')
    save('blendshapes', dataset, ids, np.stack(X))


# -- lbph: OpenCV's classic Local Binary Pattern Histogram face recognizer -- the
#    pre-deep-learning face-recognition algorithm, no CNN/learning at all. We train it
#    on the whole dataset (labels are just row indices) purely to pull out its internal
#    per-image histogram via getHistograms() as a fixed-length embedding.

def run_lbph(dataset):
    import cv2
    ids, imgs = [], []
    for iid, p in iter_images(dataset, 'data/aligned112'):
        imgs.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
        ids.append(iid)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(imgs, np.arange(len(imgs)))
    X = np.stack([h.flatten() for h in recognizer.getHistograms()])
    save('lbph', dataset, ids, X)


# -- fisherface: OpenCV's classic LDA (Fisher discriminant) face recognizer -- unlike
#    every other hand-crafted family here, this one IS explicitly trained/fit to
#    maximize between-identity separation (Fisher's discriminant ratio, the same idea
#    as d') over within-identity variance. It's the classical, non-deep analog of what
#    ArcFace/CosFace/AdaFace do with a CNN: engineered for high identity discriminability.
#    London is the only dataset with repeat identities, so the PCA+LDA projection is
#    fit once on all 1020 London images (102 identities), then applied (frozen, no
#    refitting) to every dataset -- exactly like a pretrained deep checkpoint, just
#    "pretrained" in-house on our own identity-labeled data instead of an external corpus.

def _fit_fisherface():
    import cv2
    imgs, labels = [], []
    label_map = {}
    for iid, p in iter_images('london', 'data/aligned112'):
        _, fn = iid.split('/')
        identity = fn.split('_')[0]
        label_map.setdefault(identity, len(label_map))
        imgs.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
        labels.append(label_map[identity])
    rec = cv2.face.FisherFaceRecognizer_create()
    rec.train(imgs, np.array(labels, dtype=np.int32))
    return rec.getMean().flatten(), rec.getEigenVectors()


def run_fisherface(dataset):
    import cv2
    mean, eigvecs = _fit_fisherface()
    ids, X = [], []
    for iid, p in iter_images(dataset, 'data/aligned112'):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        flat = img.flatten().astype(np.float64)
        X.append((flat - mean) @ eigvecs)
        ids.append(iid)
    save('fisherface', dataset, ids, np.stack(X))


RUNNERS = dict(facenet=run_facenet, arcface=run_arcface, cosface=run_cosface,
               adaface=run_adaface, fairface=run_fairface, geometric=run_geometric,
               clip=run_clip, dinov2=run_dinov2, blendshapes=run_blendshapes,
               lbph=run_lbph, fisherface=run_fisherface)

if __name__ == '__main__':
    fam = sys.argv[1]
    targets = [sys.argv[2]] if len(sys.argv) > 2 else DATASETS
    for ds in targets:
        out = os.path.join(EMB, f'{fam}__{ds}.npz')
        if os.path.exists(out):
            print(f'skip existing {out}')
            continue
        RUNNERS[fam](ds)
