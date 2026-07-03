# Definitions

Rigorous definitions of every term used in this study's figures and results. Read this
before interpreting any number in `presentables/` or `results/`.

---

## 1. Face embedding

A **face embedding** is a fixed-length vector of numbers produced by feeding a face
photo through a frozen (never fine-tuned in this study) model. Each of the 7 families
below is an *independent* pipeline — they do not share a common "raw" feature vector;
each was trained separately, on different objectives, and each throws away whatever its
own training objective didn't need. There is no universal "true" feature vector of a
face that these are all approximations of — each embedding is that specific model's own
lossy summary, shaped by its own training data and loss function.

| Family | Architecture | Training objective | Training data | Dimensions |
|---|---|---|---|---|
| FaceNet | InceptionResnetV1 (CNN) | Triplet loss — pull same-identity pairs together, push different-identity pairs apart, in Euclidean space | VGGFace2 | 512 |
| ArcFace | ResNet-50 (CNN) | Additive angular margin — same idea, enforced as an angular margin on a hypersphere | WebFace600K | 512 |
| CosFace | ResNet-100 (CNN) | Additive cosine margin (m=0.4) — same idea, margin applied to cosine similarity | Glint360K | 512 |
| AdaFace | IR-101 (CNN) | Quality-adaptive margin — margin strength adjusts per-image based on estimated image quality/pose, explicitly built for robustness to messy/varied-pose photos | WebFace4M | 512 |
| FairFace | ResNet-34 (CNN), penultimate layer | Attribute classification — predict race/gender/age, **never trained to distinguish individuals** | FairFace | 512 |
| Geometric | none (hand-crafted) | No learning — ratios/symmetry measures computed directly from 106 detected landmark points (eye spacing, jaw width, etc.) | — | 221 |
| CLIP | ViT-L/14 (Vision Transformer, not a CNN) | Image–text contrastive — match photos to captions; not face-specific at all | WebImageText | 768 |

**"Trained for identity"** = FaceNet, ArcFace, CosFace, AdaFace (their explicit job is
telling people apart). **"Trained, not for identity"** = FairFace, CLIP (they learned
something about faces/images, but never "is this the same person"). **"Hand-crafted
(untrained)"** = Geometric (no learning process at all — a fixed formula).

---

## 2. Attractiveness rating (the prediction target, y)

Every dataset's rating is **the mean of many human raters' scores for one photo** — not
a single person's opinion, and not an objective property of the face. Ratings differ
across datasets in scale, rater-pool size, and (for MEBeauty) exact construction:

| Dataset | Scale | Raters | Verified how? |
|---|---|---|---|
| SCUT-FBP5500 | 1–5 | Fixed panel of **60 raters**, every rater scored every image | Confirmed from the dataset's own `README.txt` |
| MEBeauty | Observed range **1.62–9.62** in the score file actually used (`train_/test_universal_scores.csv`) | "About 300 individuals" total rater pool (per the dataset's README) — **the exact number of raters per individual image, and the precise formula used to build the "universal" score from the dataset's separate "generic" and "date" scores, are not documented in the source repository.** We used the file the dataset's own authors designated as their default train/test target. | Score range verified directly from the CSV; rater-count-per-image and "universal" construction are an acknowledged gap, not fabricated |
| London Set | 1–7 | **2,513 raters, every one of whom rated every one of the 102 identities' neutral-front photo** (verified by counting non-empty cells in the ratings file — min = max = mean = 2,513) | Directly counted from `london_faces_ratings.csv` |

**Why this matters, and why it's not just pedantry:** the "ground truth" a model is
being scored against is itself an *estimate* — the mean of a finite, noisy panel of
human opinions, not a Platonic attractiveness value. Averaging more raters produces a
more stable, lower-noise estimate of "consensus attractiveness." London's target is
therefore the **most reliable of the three by rater count** (2,513 vs. 60 vs. an
unknown-but-likely-smaller MEBeauty count) — yet London produced the *lowest* accuracy
numbers across every embedding. That tells us the low London accuracy is **not** caused
by a noisy target; it's almost certainly driven by London's tiny sample size (102 images
total, so 5-fold cross-validation trains on ~82 and tests on ~20 at a time) — there is
simply much less data for the ridge probe to learn the score-predicting relationship
from, and the resulting accuracy estimate is itself statistically noisier (small-n
correlations swing more from fold to fold). **These are two different kinds of
"reliability," and they can point in opposite directions for the same dataset:**

- **Target/label reliability** — how much averaging-out-of-noise happened in computing
  y itself (more raters → cleaner target). London wins here.
- **Estimation reliability of the accuracy number** — how much data the probe had to
  learn from and how stable the resulting r is. London loses badly here (n=102 vs.
  thousands).

---

## 3. Prediction accuracy — precise definition

We train a **ridge regression probe**: `StandardScaler` (each embedding dimension
rescaled to zero mean/unit variance) followed by `Ridge` regression (linear regression
with an L2 penalty; its strength `alpha` is chosen per fold by an inner 3-fold
cross-validation grid search). This same probe architecture is used, unmodified, for
every embedding family and every dataset — the only thing that changes is which
embedding vectors go in. This is deliberate: it isolates *what the frozen embedding
encodes*, not how flexible or well-tuned the downstream predictor is. (A more flexible
256-unit MLP was tried and dropped — it lost to ridge on every family/dataset, badly on
London, indicating overfitting rather than real non-linear signal ridge was missing.)

Given true ratings `y` and the probe's predictions `ŷ`, "accuracy" is reported as three
numbers, always computed together:

- **Pearson r** — the standard linear correlation coefficient:
  `r = cov(y, ŷ) / (std(y) · std(ŷ))`, range [−1, 1]. Measures how well `ŷ` tracks `y`
  as a *straight-line* relationship. **This is the headline "accuracy" number quoted
  everywhere in this study** ("CLIP accuracy = 0.84" means Pearson r = 0.84).
- **Spearman rho** — the same idea, computed on the *rank order* of y and ŷ rather than
  their raw values. Less sensitive to outliers and to whether the relationship is
  exactly linear vs. just monotonic. Reported alongside Pearson as a robustness check;
  the two rarely disagree by much in this study, which is itself a check that the
  relationship is close to linear.
- **RMSE** (root-mean-squared error) — reported in *z-scored* units for cross-dataset
  comparisons (since raw scores live on different scales — SCUT's 1–5 vs. MEBeauty's
  ~1.6–9.62 vs. London's 1–7 — comparing raw RMSE across datasets would be meaningless;
  z-scoring puts every dataset's target on the same standardized footing before
  computing this one metric).

### Is "accuracy" (Pearson r) actually comparable across datasets?

**Partially — precisely because of what it does and doesn't correct for:**

- **Yes, in the sense that matters for interpretation.** Pearson r is invariant to any
  linear rescaling of `y` — computing r on SCUT's 1–5 scale, MEBeauty's ~1–10 scale, and
  London's 1–7 scale is legitimate; the different units don't need correcting for. r=0.84
  means the same thing statistically on every dataset: the model's predictions explain
  r²=71% of the variance in the (rescaled) ratings.
- **No, in the sense that the *reliability of that r estimate*, and what a "perfect"
  score of r=1 would even be achievable, differ by dataset.** A dataset's **noise
  ceiling** — the maximum r any model could achieve, limited by how much random,
  unmodelable disagreement exists between individual human raters — depends on how
  reliable the mean rating itself is (rater count) and on unknown between-dataset
  differences in rating protocol/population. We do not have the individual-rater data
  needed to compute an exact noise ceiling for each dataset here (this is a known
  limitation, not something this study resolves) — but the London findings above make
  clear that raw accuracy numbers should not be read as directly comparable "how good is
  this embedding" scores without accounting for sample size and (unverified) rater
  reliability per dataset.

**Bottom line:** treat within-dataset accuracy comparisons *across embedding families on
the same dataset* as the most trustworthy reads (same target, same n, same protocol —
only the embedding changes). Treat *across-dataset* accuracy comparisons (e.g. "CLIP is
better on SCUT than on London") more cautiously — some of that gap is genuine embedding
weakness, and some of it is London's much smaller sample size and different, less
documented rating protocol.

---

## 4. Within-dataset vs. cross-dataset accuracy

- **Within-dataset**: 5-fold cross-validation *inside one dataset* — train the probe on
  80% of that dataset's images, test on the held-out 20%, rotate 5 times, report the
  accuracy metrics on the pooled held-out predictions. Tests: "does this embedding
  predict attractiveness well for faces similar to the ones it was calibrated on?"
- **Cross-dataset**: train the probe on *all* of dataset A, test it (with no
  retraining) on *all* of dataset B — for every ordered pair among the 3 datasets (6
  pairs total). Scores are z-scored per dataset first (mean 0, SD 1) so a probe trained
  on one scale can sensibly predict on another. Tests: "does what this embedding
  captures about attractiveness generalize to a different population, camera setup, and
  rating protocol?" — a much harder and more meaningful test of whether the embedding
  found something real rather than something dataset-specific.

---

## 5. Pose-robust identity discriminability (d′)

Measured **only** on the London Set, because it is the only dataset with multiple
photos of the same identity (102 identities × 5 poses × 2 expressions = 10 photos per
person). This is a **signal-detection** measure, computed on L2-normalized embeddings:

1. For each identity, compute the cosine similarity between their frontal photo's
   embedding and each of their other neutral-pose photos' embeddings (front vs.
   3-quarter, front vs. profile) → the **same-identity, cross-pose** similarity
   distribution.
2. Compute the cosine similarity between every pair of *different* identities' frontal
   photos → the **different-identity, same-pose** similarity distribution.
3. `d′ = (mean(same-identity) − mean(different-identity)) / pooled standard deviation`

**Interpretation:** d′ measures how confidently an embedding can tell "this is the same
person from a different angle" apart from "this is simply a different person." Higher
d′ = the embedding locks onto identity more strongly regardless of pose (more
pose-invariant, in the specific sense that matters for face-verification robustness).
**A negative d′** (as observed for the Geometric baseline, d′ = −2.87) means the
representation is *dominated by pose*: two photos of the *same* person from different
angles look less similar, on average, than two *different* people's frontal photos —
turning your head moves the raw landmark ratios more than being a different person does.

---

## 6. What "the tradeoff" claim actually means

The study's central claim is a **correlation across 7 data points** (one per embedding
family) between each family's d′ (above) and its mean prediction accuracy (Pearson r,
averaged across datasets). With only 6–7 families, this is a small-sample correlation —
reported honestly at p≈0.06–0.07, suggestive and mechanistically consistent, not a
tight statistical guarantee. See `results/invariance.csv` for the exact per-family
numbers and `RESULTS.md` (repository root) for the full write-up, caveats, and
limitations.

---

## Files in this folder

- `methodology_flowchart.png` — the full pipeline, raw photo to final correlation
- `accuracy_by_family.png` — each embedding's within-dataset accuracy, broken out by dataset
- `invariance_and_accuracy_bars.png` — d′ and accuracy side by side, same ordering, color-coded by training objective
- `dprime_vs_accuracy.png` — the headline scatter: d′ vs. accuracy, with the fitted trend line
