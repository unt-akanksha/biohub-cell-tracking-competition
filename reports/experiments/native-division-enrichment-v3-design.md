# Division-event enrichment: next evidence-driven experiment

## Measured gap

The native CNN/transformer pair generalizes in conditional parent association,
but two independent parent probabilities plus persistence failed as a division
detector: five scored false branches, no recovered true branches. That candidate
is closed. Do not retrain on its four target movies or select its observed false
cases as negatives, and do not weaken its thresholds to try again.

Read-only inspection of the same91optimization/10selection GEFFs, with all21
files per movie hash-verified, found a correctable sampling problem:

| Role | Movies | Movies containing divisions | All annotated division events | In native v2 | Extra event transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44b6 optimization | 27 | 14 | 19 | 2 | 17 |
| 6bba optimization | 64 | 49 | 93 | 17 | 76 |
| 44b6 existing selection | 3 | 3 | 3 | 3 | 0 |
| 6bba existing selection | 7 | 7 | 14 | 14 | 0 |

Thus112available optimization division events, not19, exist without adding
movies, opening the sealed audit, or changing selection roles. These counts are
annotations, not yet usable image-proposal triplets. Image-match recall must be
measured honestly before any head training. The93additional optimization event
transitions are listed in `native-division-coverage-v3.json`, SHA
`5e674aad7607930cbb2f0e96d204d8bd13931a6fd3f407ddea1bc8d0e49f8cbe`.

## Data work (original protocol; now completed)

Enrich optimization sampling to include **every annotated division transition**,
using the same normalized native images, physical scales and image-only proposal
centers. Preserve all existing held-out/excluded movie boundaries. Add no new
selection movies/time points: existing selection already covers all events.

Prepare explicit parent/two-daughter image examples. A positive requires an
annotated parent with two annotated adjacent-frame daughters and distinct matched
image proposals for all three. Negatives must be supported by known parentage:
one proposed daughter has a different uniquely annotated parent. Never label an
unannotated daughter or an ambiguous near match as a negative. Report available,
matched, retained and omitted positive/negative counts by movie/embryo.

Existing native v2 patches may supply additional optimization negatives only
when GT identities can be recovered unambiguously and reverified against the
frozen movie plan. Do not guess identities from patch indices. Prefer explicit
identity fields in newly prepared packets and unique verified geometry matching
for legacy packets. Preserve source-only initialization throughout reciprocal
embryo evaluation; do not use a target-trained encoder and call it embryo-held-out.

Only after triplet coverage is known, freeze a small division-specific joint
head and source-only selection/calibration protocol over the reusable native
encoders. Test functionality, class handling, symmetry under daughter exchange,
checkpoint reload and throughput before a longer fit. No GPU job is currently
queued for v3; do not describe this design as running training.

The endpoint/parent-affinity components alone did not establish a stronger
submission. Full-movie patched metrics, both embryos, worst-movie behavior and
offline runtime remain mandatory; the overall top-five objective is unproven.

## Executed data result, September 14

The four-transition functionality smoke passed in 2.409 seconds, including an
image-matched division. The full 93-transition enrichment completed in 38.872
seconds on Antelume. It added 38 usable positives, not 93. Merging disjoint
legacy packets produces 718 packets and 3,087 triplets:

| Embryo/role | Positive | Negative |
| --- | ---: | ---: |
| 44b6 optimization | 6 | 235 |
| 6bba optimization | 38 | 2,070 |
| 44b6 selection | 1 | 151 |
| 6bba selection | 6 | 580 |

Thus only 44/112 optimization annotations yield safe image-proposal triplets.
Selection contains just seven usable positive events: evidence is necessarily
weak, particularly for 44b6. No target-pilot labels or sealed audit were used.
The merged DATA.json SHA-256 is
`7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06`.

## Frozen head experiment, before fitting any v3 head

Use each embryo's own native v2 CNN-origin and transformer-origin encoders as
frozen feature extractors. These encoders have the same residual architecture
but independently learned weights; the old parent heads and transformer context
are discarded. This is NOT promotion of either failed source44 parent model.
For a reciprocal fold, only encoders trained on its source embryo are allowed.

Joint-event features are invariant to daughter order: six physical geometry
features, 24 brightness/morphology features, and 1,280 symmetric embedding
features per encoder. Fit two independent linear division classifiers per
source, each using 30 non-embedding features and its own 1,280 embedding
features. Fit a six-feature geometry-only control on the identical split.
There is no architecture/seed/regularization search: source-optimization mean
and standard deviation (floor 0.1), standardized features clipped to [-10,10],
zero initialization, 2,000 full-batch Adam steps, learning rate 0.01, balanced
positive/negative BCE plus 0.01 times the sum of squared weights, no bias
penalty. Use the final step, not selection-based early stopping.

Calibrate each head's threshold once, using only its source selection negatives:
the next float32 above the largest negative probability. Require zero source
false positives, at least 50% positive recall, and balanced NLL lower than the
geometry control. Only source-passing heads may open their opposite-embryo
diagnostic, at the unchanged threshold. The opposite gate is also zero false
positives, >=50% recall, and lower balanced NLL than the source-trained geometry
control. A half/half probability ensemble is permitted only if BOTH individual
heads pass both gates; its threshold is calibrated on source selection only.
These are screening gates, not a generalization guarantee with seven positives.

Before full feature extraction: synthetic geometry, daughter permutation,
normalization shape, finite features, exact model reload and a fixed source
packet smoke. All weights/data/source files are hash-bound. Extraction has a
600-second cap; classifier fitting has a 300-second cap. Run sequentially and
reject occupied GPUs. No automatic target graph editing or Kaggle submission.
If the screening gates fail, close this candidate without threshold tuning on
target results. A pass still requires complete-movie official scoring and runtime.
