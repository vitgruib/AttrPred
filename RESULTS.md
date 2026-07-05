# Do Face-Recognition Embeddings Differ in How Well They Predict Attractiveness?

**Research question.** Do face-embedding families that differ in pose/angle invariance
differ systematically in how well they predict human-rated facial attractiveness? The
hypothesis: higher pose invariance correlates with *lower* attractiveness-prediction
accuracy, because invariance training discards appearance variation that also carries
attractiveness signal. **A second, independent question:** for each embedding family, is
attractiveness-prediction accuracy consistent across demographic subgroups (gender,
ethnicity, and — where available — age), or does some subgroup get systematically
better/worse predictions? See "Demographic bias" under Results.

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
| DINOv2 | timm `vit_small_patch14_dinov2.lvd142m` | self-supervised (DINO distillation), no labels at all | LVD-142M | 384 |
| Blendshapes | MediaPipe Face Landmarker v2 | supervised regression to 52 named ARKit blendshape coefficients | Google's internal face-landmark corpus | 52 |
| LBPH | OpenCV `cv2.face.LBPHFaceRecognizer` | hand-crafted, no learning (grid of Local Binary Pattern histograms) | — | 16384 |
| Fisherface | OpenCV `cv2.face.FisherFaceRecognizer` (PCA+LDA) | supervised linear discriminant — explicitly fit to *maximize* between-identity separation over within-identity (pose) variance | fit in-house on the London Set's own 102 identities (1,020 images) | 101 |

FairFace is the non-identity contrast class (trained to *keep* visible attributes, not
discard them for identity); Geometric and LBPH are the two classic non-deep baselines
(hand-crafted features, zero learned parameters); CLIP and DINOv2 are non-face-specific
foundation-model baselines — CLIP contrastive against text, DINOv2 self-supervised with
no supervision signal whatsoever; Blendshapes is a trained-but-non-identity model, like
FairFace, except its 52 dimensions are individually named, interpretable local
expression/shape coefficients rather than a black-box feature vector. Fisherface is the
classical (non-deep) counterpart of the identity-margin CNNs: instead of a neural network
trained on millions of external identities, it's a linear PCA+LDA projection fit directly
on our own only identity-labeled corpus (London's 102 people × 10 views), explicitly
optimized — by the textbook Fisher discriminant criterion — to separate identities as
strongly as the data allows, pose variation included. It answers "does a classical
method engineered for maximal identity discriminability behave like the deep
margin-based ones?", using a completely different mechanism (linear projection vs. deep
metric learning) than any other family here.

**Datasets (3, all with human attractiveness ratings; cross-dataset generalization is a core requirement):**

| Dataset | n (rated) | scale | notes |
|---|---|---|---|
| SCUT-FBP5500 | 5,500 | 1–5 | frontal, Asian/Caucasian × M/F, mean of 60 raters |
| MEBeauty | 2,197 | 1–10 | in-the-wild, 6 ethnic groups |
| Face Research Lab London | 102 | 1–7 | rated on neutral-front; 10 views/identity retained for the invariance probe |

**Pipeline.** Single detector/aligner for all datasets (insightface RetinaFace →
5-point ArcFace alignment to 112×112 for the identity models, LBPH, and Fisherface;
loose 1.3× crop at 224 for CLIP/FairFace/DINOv2/Blendshapes). Frame-filling faces that
RetinaFace missed at the image edge were recovered with a reflect-padded-border retry
(final detection failure: 1 / 8,718 images). MediaPipe's own internal face detector
(used only for Blendshapes) is a separate, lighter model tuned for frontal/near-frontal
faces; it fails much more often on the loose 224 crop when the face is a true profile
(see Limitations). Fisherface's PCA+LDA projection is fit once (grayscale 112×112,
London's 1,020 images / 102 identity labels) and then applied frozen to all three
datasets, exactly like a pretrained deep checkpoint — just "pretrained" in-house on our
own identity-labeled data instead of an external corpus. Each frozen
backbone → an identical lightweight ridge regression probe on top (alpha tuned by inner
3-fold CV), kept deliberately linear and identical across families so the comparison
isolates what each embedding encodes rather than how much capacity the downstream
predictor has. (A 256-unit MLP was tried as a non-linear alternative and dropped: it
lost to ridge on every family/dataset — badly on the small London set (n=102), where
several MLP scores went to ~0 or negative — indicating overfitting rather than real
non-linear signal ridge was missing.) Within-dataset: 10-fold CV, Pearson/Spearman of
predicted vs. true score. Cross-dataset: train on all of A, test on all of B (scores
z-scored per dataset), all ordered pairs.

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
in the literature. The headline correlation uses the 9 *trained* embeddings
(FaceNet/ArcFace/CosFace/AdaFace/FairFace/CLIP/DINOv2/Blendshapes/Fisherface); the two
hand-crafted baselines (Geometric, LBPH) never underwent any training process at all,
let alone one that traded off pose-invariance for anything, so they are reported
separately rather than folded into a trend about trained-model tradeoffs.

**Important asymmetry in how d′ is computed for Fisherface vs. every other family.**
Every other embedding's d′ is a genuinely zero-shot measurement: the model was trained
on some external corpus that has nothing to do with the London Set, and we then measure
how well it happens to separate London's 102 identities. Fisherface's PCA+LDA projection
was fit *directly on those same 102 identities* (with identity labels, though not
attractiveness labels) — so its d′ is not a zero-shot generalization number, it's closer
to a training-set fit score. Treat Fisherface's d′ as an upper-bound/best-case
demonstration of "what maximal engineered identity-discriminability looks like," not as
directly comparable to the other families' out-of-sample d′ values. Its
attractiveness-accuracy numbers on SCUT/MEBeauty are unaffected by this (those images
were never seen while fitting the projection); only its London within-dataset accuracy
number shares images with the Fisherface fit (again, only via identity labels, not
attractiveness scores) and should be read with that in mind.

**By-category analysis.** In addition to the headline correlation over all trained
families, we also analyze the four natural (deep vs. classical) × (trained-for-identity
vs. not) categories separately, then compare them as groups — see "Results by category"
below.

**Demographic bias measure.** Separately from the invariance question, we ask whether
each family's attractiveness-prediction accuracy is *consistent across demographic
subgroups*, using the same 10-fold ridge probe's out-of-fold predictions. For each
family/dataset, we compute Pearson r within each subgroup (gender, ethnicity; a subgroup
must have ≥20 rated images to be reported at all — below that, r is too noisy to trust)
and report the **gap** = highest-subgroup r − lowest-subgroup r as the headline number:
a small gap means the embedding predicts attractiveness about equally well regardless of
who's in the photo; a large gap means accuracy depends on which demographic group the
face belongs to. Ethnicity granularity differs by dataset (SCUT: asian/caucasian only;
MEBeauty: 5 groups; London: only the "white" subgroup clears the ≥20 floor — the other 4
London ethnicity groups have n=9–13 each) — gaps are therefore only ever compared within
one dataset, never pooled across datasets' different group definitions. **Age** has no
ground-truth label in SCUT or MEBeauty at all (see Method above); age-subgroup analysis
uses London's real per-identity age (18–54, n=100/102) split into runtime tertiles
(`<=24`, `25-29`, `>29`, ~30–42 images each). We did not estimate age for SCUT/MEBeauty
with a separate pretrained model — that would introduce a second, noisier learned
attribute as the analysis's own confound, trading one open question for another.

## Results

### Within-dataset accuracy (ridge probe, 10-fold CV Pearson r)

| Family | SCUT (n=5500) | MEBeauty (n=2197) | London (n=102) | **mean** |
|---|---|---|---|---|
| **CLIP** | **0.938** | **0.826** | **0.805** | **0.856** |
| DINOv2 | 0.878 | 0.730 | 0.700 | 0.769 |
| FaceNet | 0.867 | 0.744 | 0.508 | 0.706 |
| CosFace | 0.872 | 0.625 | 0.534 | 0.677 |
| FairFace (attr) | 0.841 | 0.690 | 0.493 | 0.675 |
| LBPH | 0.777 | 0.564 | 0.568 | 0.636 |
| Geometric | 0.774 | 0.559 | 0.405 | 0.579 |
| AdaFace | 0.854 | 0.524 | 0.270 | 0.549 |
| ArcFace | 0.828 | 0.499 | 0.249 | 0.525 |
| Blendshapes | 0.553 | 0.356 | 0.049 | 0.320 |
| Fisherface | 0.634 | 0.414 | **−0.099** | 0.317 |

Fisherface is the *worst* family overall — the only one with a negative accuracy
(London) — but by a razor-thin margin over Blendshapes (0.317 vs. 0.320 mean); treat
these two as effectively tied for last rather than a clean ranking.

### Cross-dataset accuracy (train A → test B, mean Pearson r over the 6 ordered pairs)

| Family | mean cross-r |
|---|---|
| CLIP | 0.703 |
| DINOv2 | 0.564 |
| FaceNet | 0.531 |
| FairFace | 0.517 |
| CosFace | 0.444 |
| Geometric | 0.375 |
| LBPH | 0.362 |
| AdaFace | 0.267 |
| ArcFace | 0.226 |
| Fisherface | 0.248 |
| Blendshapes | 0.148 |

Ordering is mostly preserved across the within/cross split: CLIP generalizes best,
DINOv2 second; Blendshapes and Fisherface generalize worst. One notable flip: FairFace
edges out CosFace cross-dataset (0.517 vs. 0.444) despite CosFace scoring higher
within-dataset (0.677 vs. 0.675) — a reminder that within-dataset and cross-dataset
accuracy are related but not identical measurements, and close calls can go either way.

### Invariance vs. accuracy

| Family | d′ (pose) | mean within-r | mean cross-r | trained? |
|---|---|---|---|---|
| DINOv2 | −0.53 | 0.769 | 0.564 | yes (self-supervised) |
| Blendshapes | 0.90 | 0.320 | 0.148 | yes (supervised, non-identity) |
| FairFace | 1.75 | 0.675 | 0.517 | yes |
| CLIP | 3.08 | 0.856 | 0.703 | yes |
| FaceNet | 4.34 | 0.706 | 0.531 | yes |
| CosFace | 7.01 | 0.677 | 0.444 | yes |
| ArcFace | 7.22 | 0.525 | 0.226 | yes |
| AdaFace | 7.28 | 0.549 | 0.267 | yes |
| LBPH | −2.45 | 0.636 | 0.362 | no (hand-crafted) |
| Geometric | −2.87 | 0.579 | 0.375 | no (hand-crafted) |
| **Fisherface** | **28.42** | 0.317 | 0.248 | yes (classical LDA, see caveat above) |

Fisherface's d′ (28.42) is roughly **4× higher** than the next-highest family
(AdaFace/ArcFace, ~7.2–7.3) — expected, since it was explicitly fit to maximize exactly
this quantity on exactly this data (see the asymmetry caveat above), not a coincidence
of architecture.

### Results by category

Instead of only pooling every trained family into one correlation, we also analyze the
four (deep vs. classical) × (identity-trained vs. not) categories on their own, then
compare them as groups:

| Category | n | mean d′ | mean within-r | mean cross-r |
|---|---|---|---|---|
| Deep, trained for identity | 4 | 6.46 | 0.615 | 0.367 |
| Deep, trained not for identity | 4 | 1.30 | 0.655 | 0.483 |
| Classical, trained for identity | 1 | 28.42 | 0.317 | 0.248 |
| Classical, hand-crafted (untrained) | 2 | −2.66 | 0.608 | 0.369 |

Within-category d′-vs-accuracy correlations (only meaningful for the 4-member groups):
**deep/identity-trained r = −0.73 (p = 0.27)** — the tradeoff still points the right
way *within* this group alone (FaceNet > CosFace > AdaFace ≈ ArcFace, tracking d′
fairly well); **deep/not-identity-trained r = +0.27 (p = 0.73)** — no relationship
within this group (DINOv2 and CLIP both do well despite very different d′; Blendshapes
does badly despite also having low d′ — capacity, not invariance, differentiates this
group, per Finding 1 below).

**The clearest single comparison in the whole study is category-vs-category, not
family-vs-family:** the two categories explicitly engineered/trained for identity
discriminability (deep margin-CNNs *and* classical Fisherface) sit at mean within-r
0.615 and 0.317 respectively — both below the two categories that were *not* trained
for identity (0.655 for deep-not-identity, 0.608 for classical-untrained). Moving from
"trained for identity" to "not trained for identity" costs accuracy within *both* the
deep and the classical mechanism — this is the tradeoff appearing as a categorical
effect (2×2 grouped comparison), independent of the noisier continuous d′-vs-accuracy
correlation across individual families. The cost is dramatic for the classical
mechanism (0.29 gap) and real but modest for the deep mechanism (0.04 gap) — the
categorical effect is consistent in direction but not equal in size across mechanisms.

`d′` = separation of same-identity cross-pose similarities from different-identity
frontal similarities — how discriminably an embedding preserves identity across pose.
Negative d′ (Geometric, LBPH, and now DINOv2) means the representation actually
separates *pose* more strongly than *identity* on average — turning your head changes
the representation more than being a different person does. For Geometric/LBPH this
reflects raw pixel/landmark measurements moving with head angle; for DINOv2 it reflects
a self-supervised model that was never given any identity signal to lock onto at all.

### Demographic bias

**Ethnicity: Black subgroup accuracy is lowest in 8 of 11 families on MEBeauty, Asian
is highest in 9 of 11.** MEBeauty is the only dataset with all 5 ethnicity groups at
usable n (caucasian 972, hispanic 293, indian 294, asian 344, black 294 rated images).
Ranking each family's 5 group-accuracies: Black is the single lowest-scoring group for
FaceNet, ArcFace, CosFace, AdaFace, FairFace, Geometric, CLIP, and DINOv2 — every deep
identity-margin model plus every deep non-identity model except Blendshapes. Asian is
the single highest-scoring group in 9 of the 11 families. This is a strikingly
consistent pattern across two nearly-disjoint mechanisms (triplet loss, angular margin,
attribute classification, image-text contrastive, self-supervised distillation) that
share nothing except being frozen embeddings trained on face corpora with well-documented
racial skew (VGGFace2, WebFace600K, Glint360K, WebFace4M, FairFace, WebImageText all
over-represent White/Asian faces relative to Black faces). The 3 exceptions — Blendshapes
(lowest: indian), LBPH (lowest: hispanic), Fisherface (lowest: caucasian) — are also the
3 lowest-accuracy families overall, where every group's r is low enough that "lowest
group" may just be reflecting a floor rather than a systematic disadvantage.
On SCUT (asian/caucasian only, 4000 vs. 1500 images), asian scores higher in 8 of 11
families, by a wide margin for the identity-margin models specifically (ArcFace +0.154,
AdaFace +0.112) — consistent with SCUT's own asian:caucasian ratio (asian is the
majority group there, unlike MEBeauty/London where white/caucasian is largest), i.e. the
advantaged group tracks each dataset's own majority-group composition, not one ethnicity
being universally favored.

**Gender: female accuracy beats male in all 11 families on MEBeauty, but it's a coin
flip on SCUT and London.** MEBeauty: every single family predicts attractiveness better
for women than men (gaps 0.02–0.21, largest for Blendshapes/ArcFace/LBPH). SCUT: female
higher in 5/11 families, male in 6/11, gaps mostly under 0.03. London: female higher in
5/11, male in 6/11, gaps larger (up to 0.28 for FairFace) but noisy at n=49/53 per group.
The gender effect is real and one-sided on MEBeauty specifically, not a universal
property of these embeddings — pointing at something about MEBeauty's own rater pool or
image composition (an in-the-wild, "beauty pageant"-style dataset) rather than the
embeddings themselves.

**Age (London only): larger gaps than gender or ethnicity, but on far smaller
subgroups.** Age-tertile gaps (`<=24` n=42, `25-29` n=27, `>29` n=31) average 0.20 across
families, vs. 0.08 (gender) and 0.10 (ethnicity) pooled over all 3 datasets. Read this
as **not yet established** rather than "age bias is the largest effect" — age is measured
on one dataset, in 3 groups of 27–42 each, vs. 250–4000 per group for gender/ethnicity;
smaller n alone inflates a Pearson-r gap, so the larger number here is at least partly a
sample-size artifact, not necessarily a larger true disparity. AdaFace has the widest
age gap (0.39: 0.085 for 25–29 vs. 0.469 for >29); Fisherface and Blendshapes are the
only families with a negative r in any age bucket (both already the weakest overall
predictors, see within-dataset table above).

**Accuracy and demographic consistency track together, loosely.** Averaging each
family's gender+ethnicity gaps and correlating against mean within-dataset accuracy:
r = −0.49 (p = 0.13) — more accurate families tend to have smaller subgroup gaps, but
not tightly (CLIP: gap 0.034, accuracy 0.856; FaceNet: gap 0.053, accuracy 0.706; AdaFace:
gap 0.126, accuracy 0.549). Folding in age's larger, noisier gaps pulls this to
r = −0.70 (p = 0.017) — a bigger jump than the trend should get credit for, given age's
small-n caveat above; treat −0.49 as the more trustworthy number and −0.70 as "consistent
with the same story, amplified by one noisier attribute," the same shape of caveat as the
Fisherface d′ outlier earlier in this document. The practical reading either way: **the
families that best isolate appearance from identity (CLIP, DINOv2, FaceNet) are also the
most demographically consistent ones** — being good at the task and being fair across
groups are not in tension here, unlike the classic fairness/accuracy tradeoff framing.

**Figure:** `results/bias_gap_by_family.png` (gap by family, ethnicity + gender),
`results/bias_ethnicity_heatmap.png` (MEBeauty accuracy by ethnicity × family).
**Tables:** `results/bias_gender.csv`, `results/bias_ethnicity.csv`,
`results/bias_age.csv`, `results/bias_summary.csv`.

## Findings

**1. The invariance/accuracy tradeoff is real, but its apparent strength swings a lot
depending on which families are in the sample — and the reasons why are the actual
finding.** Restricted to the original 6 identity/identity-adjacent families
(FaceNet/ArcFace/CosFace/AdaFace/FairFace/CLIP), the correlation between `d′` and
accuracy was moderate-to-strong: within-dataset Pearson r = −0.68 (p = 0.14). Adding
DINOv2 and Blendshapes (8 trained families) collapsed it to essentially nothing
(r = −0.11, p = 0.79). Adding Fisherface on top (9 trained families) swings it back to
**r = −0.56 (p = 0.12)** — but Spearman rank correlation only moves to
**rho = −0.48 (p = 0.19)**, a smaller jump than Pearson's, which is the signature of
one high-leverage outlier doing a lot of the work rather than a genuine shift in the
overall pattern: Fisherface's d′ (28.42) is ~4× any other family's, so it pulls a linear
fit hard. Given the honest caveat that Fisherface's d′ isn't measured zero-shot like the
others (see Method), **the safest reading of the pooled correlation is "consistent with
the hypothesis, not tightly significant, and sensitive to exactly which families you
include"** — which is itself informative: a real effect that's this fragile to sample
composition needs the *mechanism*-level evidence below more than the single pooled r
number.
  - **DINOv2 supports the hypothesis as strongly as any family tested, with no caveats.**
    Lowest zero-shot d′ of any trained embedding (−0.53), second-best predictor
    (0.769 within, 0.564 cross), right behind CLIP — from a training paradigm
    (self-distillation, zero labels) completely unlike any of the original 6 families.
  - **Fisherface supports the hypothesis more dramatically than any family tested, but
    with the caveat above.** Explicitly engineered for maximal identity discriminability
    (d′ = 28.42 — 4× the next-highest family) and is simultaneously the *worst*
    attractiveness predictor of all 11 families (0.317 within, negative on London). This
    is the classical-statistics mirror of the CNN margin-model result: whether you
    maximize identity separation with a deep network or a 1936-vintage linear
    discriminant, forcing it as hard as the data allows costs you the most accuracy of
    anything tested.
  - **Blendshapes breaks the simple monotone story in the other direction.** Low d′
    (0.90, same range as FairFace) yet effectively tied with Fisherface for the *worst*
    predictor (0.320 within) — because `d′` only captures whether an embedding discards
    appearance for pose-invariance, not whether it retains enough *capacity* to encode
    appearance at all. Blendshapes' 52 hand-named scalars are too low-dimensional to
    carry much appearance signal regardless of what they were or weren't trained to
    discard. **Low d′ is necessary but not sufficient** for good accuracy, and
    Fisherface/Blendshapes bracket the story from opposite sides: one shows what
    happens when you push d′ up as far as possible (accuracy craters), the other shows
    that pushing d′ down isn't enough on its own if capacity is too limited.

**The by-category comparison (see "Results by category") is the cleaner way to see this
same effect without relying on one continuous correlation number.** Both mechanisms that
were explicitly trained/fit for identity — deep margin-CNNs (mean within-r 0.615) and
classical Fisherface (0.317) — underperform their non-identity-trained counterparts
within the same mechanism (deep-not-identity 0.655; classical-untrained 0.608). That's a
consistent categorical effect across two completely different modeling paradigms, and
it doesn't depend on treating d′ as a continuous variable or worrying about any one
family's leverage on a regression line — though the size of the effect differs sharply
by mechanism (a 0.29 gap for classical, a more modest 0.04 gap for deep).

**Geometric and LBPH are excluded from the headline correlation on principle, not
convenience.** Both are hand-crafted, zero-learning baselines (grid LBP histograms;
landmark ratios) — their negative d′ (−2.45, −2.87) reflects raw pixel/landmark
measurements moving with head angle, not a *learned* invariance strategy that traded
identity-robustness for anything. Folding all 11 families in drops the correlation to
r = −0.51 (within, p = 0.11) because two more data points are added that were never
part of the causal story about what *training* trades away.

**2. CLIP still leads, but DINOv2 shows the effect isn't about text supervision.**
CLIP remains the single best attractiveness predictor on every dataset, within (0.86)
and across (0.70). DINOv2 — self-supervised, no text, no labels, no identity signal at
all — is a close second (0.77 / 0.56), beating every dedicated face-recognition
embedding including FaceNet. This narrows down *why* foundation models win: it isn't
specifically CLIP's image-text contrastive objective that helps, since DINOv2 gets most
of the way there with pure self-supervision. What both share, and what every
identity-margin model lacks, is simply never being trained to throw appearance away.
CLIP's edge over DINOv2 suggests language-alignment adds *something* further on top of
that baseline, but the bulk of the effect is "not identity-trained," not "trained with
text."

**3. Loss family matters more than "deep vs. hand-crafted" — and LBPH shows this most
starkly.** Among identity models, triplet-loss FaceNet (0.706) beats two of the three
angular-margin models outright (ArcFace 0.525, AdaFace 0.549) though CosFace (0.677)
closes most of that gap. The attribute-trained FairFace (0.675) and CLIP/DINOv2 support
the "not-optimized-to-discard-appearance predicts attractiveness better" reading.
**LBPH — a 19th-century-simple grid of Local Binary Pattern histograms, zero learned
parameters, zero deep learning of any kind — reaches 0.636 within-dataset, beating both
ArcFace (0.525) and AdaFace (0.549), and the Geometric landmark baseline (0.579).** A
hand-crafted texture descriptor from 2006 out-predicting 2019-era deep face-recognition
networks on this specific task is a direct illustration of the paper's thesis: what
matters is whether appearance was discarded, not how sophisticated the model that
discarded it was.

**4. Fisherface is the cleanest single confirmation of the thesis in the whole study —
with one honest asterisk.** It is worse than (or, on this metric, statistically tied
with Blendshapes for worse than) every other family, including the angular-margin CNNs
explicitly built to discard appearance for identity, and it gets there through a
completely different mechanism (linear PCA+LDA, not a deep network, not even trained on
external data). That a classical statistical method reaches the same qualitative
outcome as a modern CNN, purely by sharing the same *objective* (maximize identity
separation), is strong evidence that the effect is about the objective, not about deep
learning specifics. The asterisk: because its d′ was measured on the same identities it
was fit to separate, its position on the far right of the scatter plot is partly a "how
extreme can you make this look" demonstration rather than a fully independent zero-shot
data point — its *accuracy* numbers are not affected by this caveat, only its d′.

**Figure:** `results/dprime_vs_accuracy.png` (the tradeoff, 9 trained families).
**Tables:** `results/within.csv`, `results/cross.csv`, `results/invariance.csv`,
`results/category_summary.csv`.

**5. Demographic accuracy gaps exist in every family, and track loosely with overall
accuracy rather than trading off against it.** On MEBeauty, Black subgroup accuracy is
the lowest of 5 ethnicity groups in 8 of 11 families — a consistent pattern across
mechanisms that share nothing except training on face corpora with documented racial
skew. Female accuracy beats male in all 11 families on MEBeauty specifically (not on
SCUT or London), pointing at something about that dataset rather than a universal
property of the embeddings. The families with the smallest demographic gaps (CLIP,
FaceNet, DINOv2) are also the most accurate ones (r = −0.49, p = 0.13 between mean
gender+ethnicity gap and mean accuracy) — a family that's good at the attractiveness task
tends to be good at it *consistently*, not accurate-on-average by being much better for
one group than another. See "Demographic bias" under Results for the full breakdown,
including the age analysis (London-only — SCUT/MEBeauty have no ground-truth age) and an
honest caveat about how much of the age-inclusive r = −0.70 correlation is leverage from
one small, noisy attribute rather than a stronger true effect.


## Limitations

- **Training-data confound.** The identity backbones were trained on different corpora
  (VGGFace2 / WebFace600K / Glint360K / WebFace4M), so cross-family differences reflect
  both loss function *and* training data; they cannot be fully separated with frozen
  public checkpoints. The ArcFace vs. CosFace contrast in particular differs in both
  margin type and dataset (WebFace600K vs. Glint360K).
- **Invariance probe is single-dataset.** Pose invariance is measured only on the London
  Set (102 identities, controlled studio poses). It has not been validated on in-the-wild
  pose variation.
- **Geometric baseline uses insightface 106-pt landmarks**, not an independent
  landmarker — a mild shared-frontend caveat with the deep identity models. MediaPipe's
  Face Landmarker (used for Blendshapes) is a genuinely independent detector/model
  family, resolving the earlier gap noted here — but see below for its own failure mode.
- **Blendshapes has a severe, pose-dependent missing-data problem.** MediaPipe's
  internal face detector is tuned for frontal/near-frontal faces: it detects 102/102 on
  neutral-front, smiling-front, and both 3-quarter views, but only 19/102 and 10/102 on
  the two true-profile views (and similarly ~11–17/102 on the smiling-profile views) —
  an 83–90% miss rate on profile. The within/cross-dataset accuracy numbers are
  unaffected (built only from the always-present neutral-front image), but the d′
  computation's same-identity/cross-pose term draws on far fewer profile comparisons
  for Blendshapes than for any other family, and the surviving profile detections are
  not a random sample of the 102 identities (likely biased toward more head-on-looking
  "profile" shots) — treat Blendshapes' d′ as noisier and less representative of true
  profile-pose behavior than the other 10 families' d′.
- **LBPH is very high-dimensional relative to sample size.** Its 16,384-d grid histogram
  is ~30× the dimensionality of most other families; on London (n=102) ridge regression's
  inner-CV solver frequently fell back to a least-squares solution for ill-conditioned
  kernel matrices (a `sklearn` warning, not an error) — LBPH's small-dataset accuracy
  numbers likely have higher variance than the other families' for this reason.
- **Fisherface's d′ is fit-on-the-evaluation-set, not zero-shot.** Its PCA+LDA
  projection is trained directly on the London Set's 102 identities — the same
  identities its d′ is then measured on (see the asymmetry callout in Method). Its d′ of
  28.42 should be read as "how extreme identity-discriminability can be pushed by
  explicit optimization," not as a value directly comparable to the other families'
  out-of-sample d′. This does not affect Fisherface's SCUT/MEBeauty attractiveness
  accuracy (unseen during fitting); it does mean its own London within-dataset accuracy
  shares images (via identity labels only, not attractiveness labels) with the fit set.
- **London attractiveness ratings are on neutral-front images only**; other views inherit
  the identity's score for the invariance analysis but are not independently rated.
- **LiveBeauty** (the largest FAP dataset) was excluded: the paper was withdrawn by its
  authors (2025-03) and the dataset was never released.
- **No ground-truth age for SCUT or MEBeauty.** Only the London Set ships a per-identity
  age label; age-subgroup bias analysis is London-only (n=100, 3 groups of 27–42) and
  therefore noisier than the gender/ethnicity analysis (250–4000 per group across all 3
  datasets). We deliberately did not estimate age with a separate pretrained model for
  SCUT/MEBeauty, since a model-estimated attribute would introduce its own error as a new
  confound in a bias analysis, rather than resolving the gap.
- **Ethnicity category granularity differs by dataset** (SCUT: 2 groups; MEBeauty: 5;
  London: 5 defined, but only "white" clears the ≥20-image floor, so London contributes
  no ethnicity gap at all — see Demographic bias under Results). Gaps are only ever
  compared within a dataset, never pooled across datasets' differing group definitions.
- **Demographic subgroup sizes are themselves unequal** (e.g. MEBeauty: caucasian n=972
  vs. black/hispanic/indian n≈294 each), so part of any accuracy gap is expected sampling
  variance from smaller subgroups, not necessarily a "true" representational disadvantage
  of equal magnitude — the gap is real as measured, but its underlying cause (training-
  data skew vs. finite-sample noise vs. some mix) is not separated out by this analysis.

## Reproducing

```
python src/build_manifest.py            # unified manifest.csv
python src/preprocess.py                # detect+align -> data/aligned112, data/crops224
./run_extract_all.sh                    # 11 families x 3 datasets -> embeddings/*.npz
./run_analysis.sh                       # ridge probe, invariance -> results/
python src/figures.py                   # summary figures -> results/*.png
python src/flowchart.py                 # pipeline diagram -> presentables/methodology_flowchart.png
```

MediaPipe's Blendshapes family additionally requires the system library `libgles2`
(`sudo apt install libgles2` on Debian/Ubuntu) and a one-time download of
`models/face_landmarker.task` (fetched automatically on first run if missing — see
`src/extract.py:run_blendshapes`).
