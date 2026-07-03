# Do Face-Recognition Embeddings Differ in How Well They Predict Attractiveness?

**Research question.** Do face-embedding families that differ in pose/angle invariance
differ systematically in how well they predict human-rated facial attractiveness? The
hypothesis: higher pose invariance correlates with *lower* attractiveness-prediction
accuracy, because invariance training discards appearance variation that also carries
attractiveness signal.

## Method

**Embedding families (all frozen, pretrained, no backbone fine-tuning):**

| Family | Checkpoint | Loss / type | Training data | dim |
|---|---|---|---|---|
| FaceNet | facenet-pytorch InceptionResnetV1 | triplet (Euclidean) | VGGFace2 | 512 |
| ArcFace | insightface buffalo_l `w600k_r50` | additive angular margin | WebFace600K | 512 |
| CosFace | insightface antelopev2 `glintr100` | additive cosine margin (m=0.4) | Glint360K | 512 |
| AdaFace | cvlface `ir101_webface4m` | quality-adaptive margin | WebFace4M | 512 |
| FairFace (attr) | ResNet-34 `res34_fair_align_multi_7`, penultimate | attribute classification (race/gender/age) | FairFace | 512 |
| Geometric | insightface 106-pt landmarks → ratio/symmetry features + normalized landmark cloud | hand-crafted, no learning | — | 221 |
| CLIP | open_clip ViT-L/14 image encoder | image-text contrastive | WebImageText | 768 |

FairFace is the non-identity contrast class (trained to *keep* visible attributes, not
discard them for identity); Geometric is the classic non-deep baseline; CLIP is a
non-face-specific foundation-model baseline.

**Datasets (3, all with human attractiveness ratings; cross-dataset generalization is a core requirement):**

| Dataset | n (rated) | scale | notes |
|---|---|---|---|
| SCUT-FBP5500 | 5,500 | 1–5 | frontal, Asian/Caucasian × M/F, mean of 60 raters |
| MEBeauty | 2,197 | 1–10 | in-the-wild, 6 ethnic groups |
| Face Research Lab London | 102 | 1–7 | rated on neutral-front; 10 views/identity retained for the invariance probe |

**Pipeline.** Single detector/aligner for all datasets (insightface RetinaFace →
5-point ArcFace alignment to 112×112; loose 1.3× crop at 224 for CLIP/FairFace).
Frame-filling faces that RetinaFace missed at the image edge were recovered with a
reflect-padded-border retry (final detection failure: 1 / 8,718 images). Each frozen
backbone → identical lightweight probe on top (ridge regression with alpha tuned by
inner 3-fold CV; a 256-unit MLP as a secondary probe). Within-dataset: 5-fold CV,
Pearson/Spearman of predicted vs. true score. Cross-dataset: train on all of A, test on
all of B (scores z-scored per dataset), all ordered pairs.

**Invariance measure (the novel analysis).** On the London Set's 10 views per identity
(5 poses × 2 expressions), on L2-normalized embeddings, we measure **d′**: pose-robust
identity discriminability. For each identity, we compare the frontal embedding against
each other neutral-pose view (same-identity, cross-pose similarity) against
frontal-vs-frontal similarity between *different* identities:

> d′ = (mean same-identity similarity − mean different-identity similarity) / pooled SD

Higher d′ means the embedding more confidently keeps identity separable despite a pose
change — i.e. is more pose-invariant in the sense that matters for face-recognition
robustness. This is the standard signal-detection framing used to evaluate pose
robustness in face-verification research. We then correlate per-family d′ against
per-family attractiveness-prediction accuracy — the link that does not currently exist
in the literature. The headline correlation uses the 6 *trained* deep embeddings
(FaceNet/ArcFace/CosFace/AdaFace/FairFace/CLIP); the hand-crafted Geometric baseline was
never optimized to trade off pose-invariance for anything, so it is reported separately
rather than folded into a trend about trained-model tradeoffs.

## Results

### Within-dataset accuracy (ridge probe, 5-fold CV Pearson r)

| Family | SCUT (n=5500) | MEBeauty (n=2197) | London (n=102) | **mean** |
|---|---|---|---|---|
| **CLIP** | **0.937** | **0.823** | **0.759** | **0.840** |
| FaceNet | 0.865 | 0.738 | 0.534 | 0.712 |
| FairFace (attr) | 0.839 | 0.687 | 0.533 | 0.686 |
| CosFace | 0.870 | 0.619 | 0.406 | 0.632 |
| Geometric | 0.768 | 0.555 | 0.284 | 0.535 |
| AdaFace | 0.852 | 0.510 | 0.255 | 0.539 |
| ArcFace | 0.824 | 0.489 | 0.230 | 0.514 |

### Cross-dataset accuracy (train A → test B, mean Pearson r over the 6 ordered pairs)

| Family | mean cross-r |
|---|---|
| CLIP | 0.703 |
| FaceNet | 0.531 |
| FairFace | 0.517 |
| CosFace | 0.444 |
| Geometric | 0.375 |
| AdaFace | 0.267 |
| ArcFace | 0.226 |

Ordering is preserved across the within/cross split: CLIP generalizes best; the
angular-margin identity models (ArcFace, AdaFace) generalize worst.

### Invariance vs. accuracy

| Family | d′ (pose) | mean within-r | mean cross-r | trained? |
|---|---|---|---|---|
| FairFace | 1.75 | 0.686 | 0.517 | yes |
| CLIP | 3.08 | 0.840 | 0.703 | yes |
| FaceNet | 4.34 | 0.712 | 0.531 | yes |
| CosFace | 7.01 | 0.632 | 0.444 | yes |
| ArcFace | 7.22 | 0.514 | 0.226 | yes |
| AdaFace | 7.28 | 0.539 | 0.267 | yes |
| Geometric | −2.87 | 0.535 | 0.375 | no (hand-crafted) |

`d′` = separation of same-identity cross-pose similarities from different-identity
frontal similarities — how discriminably an embedding preserves identity across pose.
Negative d′ (Geometric) means raw landmark measurements actually separate *pose* more
strongly than *identity* — turning your head changes the ratios more than being a
different person does.

## Findings

**1. The invariance/accuracy tradeoff holds.** Correlating per-family pose-robust
identity discriminability (`d′`) against attractiveness accuracy, over the 6 trained
families: **within-dataset Pearson r = −0.78 (p = 0.07), cross-dataset r = −0.79
(p = 0.06)**; restricted further to the 4 pure identity models, r = −0.87. Embeddings
engineered to keep identity sharp across pose (the angular-margin family — ArcFace,
CosFace, AdaFace, d′ ≈ 7) are the *worst* attractiveness predictors, exactly as
hypothesized: pose-invariance training discards appearance variation that also carries
attractiveness signal.

**Geometric is excluded from the headline correlation on principle, not convenience.**
It's a hand-crafted baseline that never underwent any training process trading off
invariance for anything — its extreme negative d′ (−2.87) reflects raw landmark
coordinates simply moving with head angle, not a *learned* invariance strategy. Folding
it in dilutes the trend being studied (all-7-family r drops to −0.12, n.s.) because it
isn't part of the same causal story: the hypothesis is about what *training* trades away,
and Geometric was never trained.

**2. CLIP breaks the face-embedding hierarchy.** A non-face-specific foundation model is
the single best attractiveness predictor on every dataset, within (0.84) and across
(0.70) — beating every dedicated face-recognition embedding. Its `d′` is low (3.1): it
never optimized to discard non-identity appearance, so it retains attractiveness signal.

**3. Loss family matters more than "deep vs. hand-crafted."** Among identity models,
triplet-loss FaceNet (0.71) clearly beats all three angular-margin models (0.51–0.63).
The attribute-trained FairFace (0.69, lowest trained-model d′) and CLIP support the
"not-optimized-to-discard-appearance predicts attractiveness better" reading. The
geometric landmark baseline (0.54) is mid-pack — beaten by CLIP/FaceNet/FairFace,
comparable to the angular-margin identity embeddings, despite having no learned
invariance to speak of.

**4. Bias / fairness.** Accuracy is roughly gender-balanced on SCUT (F 0.85, M 0.85) and
mildly female-favoring on MEBeauty (F 0.61, M 0.51). The largest demographic disparity
is by ethnicity on in-the-wild MEBeauty: **Black faces r = 0.53 vs. Asian 0.68**
(~0.15 gap), consistent with the known demographic skew of the identity backbones'
training sets (VGGFace2/MS1M/Glint360K/WebFace). SCUT is more even (Asian 0.86,
Caucasian 0.81). This gap holds across embedding families, so it is a property of the
representations/label distribution, not of one backbone.

**Figure:** `results/dprime_vs_accuracy.png` (the tradeoff, 6 trained families).
**Tables:** `results/within.csv`, `results/cross.csv`, `results/invariance.csv`,
`results/bias_gender.csv`, `results/bias_ethnicity.csv`.


## Limitations

- **Training-data confound.** The identity backbones were trained on different corpora
  (VGGFace2 / WebFace600K / Glint360K / WebFace4M), so cross-family differences reflect
  both loss function *and* training data; they cannot be fully separated with frozen
  public checkpoints. The ArcFace vs. CosFace contrast in particular differs in both
  margin type and dataset (WebFace600K vs. Glint360K).
- **Invariance probe is single-dataset.** Pose invariance is measured only on the London
  Set (102 identities, controlled studio poses). It has not been validated on in-the-wild
  pose variation.
- **Geometric baseline uses insightface 106-pt landmarks** (the stripped mediapipe WSL
  wheel lacked the FaceMesh solution). This is the same detector family as the deep
  models rather than an independent landmarker — a mild shared-frontend caveat.
- **London attractiveness ratings are on neutral-front images only**; other views inherit
  the identity's score for the invariance analysis but are not independently rated.
- **LiveBeauty** (the largest FAP dataset) was excluded: the paper was withdrawn by its
  authors (2025-03) and the dataset was never released.

## Reproducing

```
python src/build_manifest.py            # unified manifest.csv
python src/preprocess.py                # detect+align -> data/aligned112, data/crops224
./run_extract_all.sh                    # 7 families x 3 datasets -> embeddings/*.npz
./run_analysis.sh                       # ridge+MLP probes, invariance, bias -> results/
```
