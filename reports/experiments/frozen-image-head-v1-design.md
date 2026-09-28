# Frozen image encoder / regularized division head v1

September 13, 2026, after the immutable image-context v2 pilot failed source
selection and BEFORE fitting this recipe. This is a distinct capacity-control
experiment, not an extension, reseed, threshold relaxation, or salvage of v2.

V2 training loss fell below 0.00001 on source 44b6 while its selected validation
BCE remained 1.0463. The source has 19 optimization positives and only two
eligible selection positives. That is consistent with overfitting; it does not
establish its cause. The other source passes, but the joint v2 gate fails.
Its target-embryo scores remain unopened and its trained weights are NOT reused.

Use the same immutable v2 source-only optimization/selection packets and exact
external-only 46,386,607-parameter microscopy warm starts. Freeze every encoder
weight. Extract existing symmetric relational features (1,283 visual/logit,
18 geometry) and 45 deterministic image-context morphology features. Morphology
uses only the existing image-derived peaks, never annotation neighborhoods.
The head has 1,346 coefficients plus intercept, not a trainable transformer.

Fit standardization on source optimization only, floor standard deviations at
1e-3, clip standardized features to +/-8, and divide each of the three blocks
by the square root of its width. No source selection or target statistics fit
the transform. Optimization example weights are the supplied quality weights,
renormalized to equal total weight in each available class/eligibility stratum.
Fit convex logistic heads with L2 penalties 0.1, 0.01, 0.001, 0.0001 using
L-BFGS-B, at most 300 iterations and analytic gradients. Unconverged fits are
ineligible. Pick one by the existing source utility (zero-FP TP, AP, -BCE), with
ties retaining the stronger penalty. The same fixed four choices apply to both
sources; no architecture, penalty or threshold search on the target embryo.

Source-selected thresholds and numerical gates are UNCHANGED: AP >=0.55 and
at least two source positives strictly above every source negative. Freeze both
source models, preprocessing and thresholds before opening opposite-embryo
scores. Transfer requires AP >=0.55 and at least one TP per target embryo,
at least three pooled TPs and at most one pooled FP at the frozen thresholds.
Otherwise reject and retain artifacts; no automatic follow-up search.

Original audit shards remain excluded. Historical project-level exposure is
acknowledged; claim model-level embryo exclusion, not a pristine audit. No
competition test, leaderboard score, or public output is used by this screen.
A pass is patch-level only: actual predicted-candidate complete-movie evidence,
patched official scoring, worst-movie checks, provenance and offline two-worker
runtime still precede submission. No failed public post-processing branch is
silently combined with this model.

Antelume only, one initially idle A10G, sequential source extraction, two CPU
threads, 45% CUDA allocator cap, 15-minute full watchdog and two-minute smoke.
The small smoke must exercise real image encoding, a real optimization-label
head fit and exact saved-head prediction replay before full launch. Preserve
immutable features, source models and terminal results; no large optimizer
checkpoint is needed for a fixed encoder and deterministic convex head. Refuse
foreign GPU processes without touching them. Reuse the verified v2 data/runtime
in place, requiring at least 128 MiB free; no dataset duplication, installations,
RSNA deletion, Kaggle GPU usage or instance shutdown.
