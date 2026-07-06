# Do Face-Recognition Embeddings Differ in How Well They Predict Attractiveness?

*A fun passion project about how good AI was at predicting attractiveness. Just a random unserious rabbit hole.*

## TL;DR

I tested 11 face-embedding families (deep identity-recognition CNNs, foundation models like
CLIP/DINOv2, and two classical non-deep baselines) against 3 human-rated attractiveness
datasets (SCUT-FBP5500, MEBeauty, Face Research Lab London Set), using an identical frozen
embedding → ridge regression probe for every family so the comparison isolates what each
embedding *encodes* rather than how good its predictor is.

**Headline findings:**

1. **The tradeoff is real, and shows up most cleanly as a category effect.** Split families
   into (deep vs. classical) × (trained for identity vs. not) — in *both* mechanisms, the
   identity-trained variant is worse at predicting attractiveness than its non-identity
   counterpart. The single pooled correlation number is fragile (swings from r=−0.68 to
   r=−0.11 to r=−0.56 depending which families are included), but the categorical comparison
   holds up.
2. **Foundation models win, and it's not about text.** CLIP is the best predictor on every
   dataset. DINOv2 — self-supervised, no text, no labels, no identity signal at all — is a
   close second, beating every dedicated face-recognition model. What they share is simply
   never being trained to discard appearance.
3. **Fisherface and Blendshapes tie for worst, for opposite reasons.** Fisherface (classical
   linear discriminant, explicitly maximizing identity separation) confirms the tradeoff
   directly. Blendshapes (52 hand-named expression values, never trained for identity) shows
   that low pose-discriminability alone isn't enough if the representation is too
   low-dimensional to carry appearance info in the first place.
4. **A 2006-vintage classical method (LBPH) beats several deep face-recognition CNNs**
   (ArcFace, AdaFace) at this task — evidence it's the training *objective*, not model
   sophistication, that matters.
5. **Demographic accuracy gaps exist in every family, and track *with* overall accuracy, not
   against it.** Black-subgroup accuracy is lowest in 8/11 families on MEBeauty. The most
   accurate families (CLIP, FaceNet, DINOv2) also have the smallest demographic gaps — no
   accuracy/fairness tradeoff shows up here.

## Read more

- [`results/REPORT.md`](results/REPORT.md) — condensed write-up with figures
- [`RESULTS.md`](RESULTS.md) — full method, every table, every caveat and limitation
- [`FEASIBILITY.md`](FEASIBILITY.md) — early scoping notes (datasets, checkpoints, what was ruled out)

## Reproducing

```
python src/build_manifest.py            # unified manifest.csv
python src/preprocess.py                # detect+align -> data/aligned112, data/crops224
./run_extract_all.sh                    # 11 families x 3 datasets -> embeddings/*.npz
./run_analysis.sh                       # ridge probe, invariance -> tables/
python src/figures.py                   # summary figures -> results/*.png
python src/flowchart.py                 # pipeline diagram -> results/methodology_flowchart.png
```

See `RESULTS.md` for the full method and `FEASIBILITY.md` for dataset/checkpoint sources.

---

