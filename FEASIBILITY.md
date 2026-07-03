# Feasibility Report — Embedding Types vs. Attractiveness Prediction

Date: 2026-07-02 · Environment: WSL2, Python 3.12.3, RTX 3060 12GB, 682GB free on /mnt/d

## Datasets

| Dataset | Status | Access | License | Notes |
|---|---|---|---|---|
| **SCUT-FBP5500** | ✅ Available now | Direct Google Drive link, ~172MB, no request form | Non-commercial research only | 5,500 frontal faces, beauty scores 1–5 + distributions + landmarks. [Repo](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) |
| **Face Research Lab London** | ✅ Available now | Direct figshare download | **CC-BY** (most permissive of the set) | 102 identities, 1350×1350. Attractiveness ratings (1–7, n=2,513 raters) for *neutral front* images. Believed to include multiple viewpoints per identity (front/three-quarter/profile) — verify on download; if true it doubles as pose-invariance probe data. [figshare](https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666) |
| **MEBeauty** | ✅ Available now — **already cloned to `data/MEBeauty`** | Full dataset is in the GitHub repo itself: `original_images/{female,male}` (2,547 images, 460MB), score spreadsheets in `scores/`, official train/val/test splits | Non-commercial research only | 6 ethnic groups — the best dataset for the bias/fairness breakdown. [Repo](https://github.com/fbplab/MEBeauty-database) |
| **LiveBeauty** | ❌ Dead end | Paper **withdrawn by authors** (2025-03-13) citing description errors; dataset was never released ("available soon" never materialized) | — | Drop from plan. [arXiv 2501.02509](https://arxiv.org/abs/2501.02509) |

**Core plan:** All three datasets (SCUT-FBP5500, MEBeauty, London Set) are immediately usable → full 3×3 within/cross-dataset evaluation possible today. LiveBeauty is out (withdrawn).

## Pretrained checkpoints

| Embedding family | Source | Status |
|---|---|---|
| **FaceNet** (triplet loss) | `facenet-pytorch` `InceptionResnetV1(pretrained='vggface2')` — auto-downloads | ✅ pip-installable, resolves on py3.12 |
| **ArcFace** (angular margin) | `insightface` `buffalo_l` pack (ONNX, auto-downloads) or `arcface_torch` model zoo | ✅ insightface 1.0.1 resolves on py3.12 |
| **CosFace** (cosine margin) | insightface `arcface_torch` model zoo — **glint360k_r100 confirmed CosFace**: `config.margin_list = (1.0, 0.0, 0.4)` | ✅ OneDrive download from [model zoo](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/README.md) |
| **AdaFace** (quality-adaptive) | HuggingFace `minchul/cvlface_adaface_ir101_webface4m` (CVLface official) | ✅ direct HF download |
| **Attribute baseline** | FairFace ResNet34 (`res34_fair_align_multi_7`) — Google Drive, fetch via `gdown` | ✅ [Repo](https://github.com/dchen236/FairFace) |
| **Geometric baseline** | MediaPipe FaceMesh (468 landmarks → ratio/symmetry features) — avoids dlib's cmake build pain on py3.12 | ✅ mediapipe 0.10.35 resolves |
| **CLIP** (stretch) | `open_clip_torch` ViT-B/32 or ViT-L/14 | ✅ resolves |

Detection/alignment: insightface bundles **RetinaFace** in `buffalo_l` — use it as the single consistent preprocessor for all backbones.

## Gaps / decisions needed

1. ~~MEBeauty requires an email request~~ — resolved: full dataset ships in the GitHub repo; cloned to `data/MEBeauty`.
2. **Pose-invariance measurement needs same-identity multi-pose images.** Options, in order of preference:
   - London Set multi-view images (if confirmed on download — likely)
   - **CPLFW** (cross-pose LFW, freely downloadable, same-identity pose pairs) as a dedicated invariance probe set
   - Continuous alternative: estimate yaw per image (e.g. from insightface pose output) and measure within-identity embedding drift vs. Δyaw
3. **Training-set overlap caveat**: all four identity models were trained on different data (VGGFace2, MS1M/Glint360K, WebFace4M) — invariance differences are partly loss-function and partly data. Worth one paragraph in the write-up; can't be fully controlled with frozen public checkpoints.
4. **Withdrawn-paper note**: LiveBeauty's absence means the largest dataset is out; the study stands on 3 datasets, all available now.

## Environment

- venv created at `.venv`; `facenet-pytorch, insightface, mediapipe, open_clip_torch` all resolve without conflict (torch 2.2.2 + CUDA 12.1 wheels, numpy 1.26).
- RTX 3060 12GB is ample for frozen-backbone inference; full embedding extraction for ~8K images × 7 models is hours, not days.
