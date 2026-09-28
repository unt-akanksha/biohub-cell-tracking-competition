# ZebraHub-to-Biohub division transfer v1

September 13, 2026. New data-transfer experiment, frozen before any fit or
selection prediction. Earlier image-context and frozen-head rejections stand.

The optimization-only GEFF audit found all 19 source-44b6 divisions already in
the image archive; only five satisfy the deployment geometry gate. There is no
extraction fix that adds independent 44b6 divisions. Source-6bba has 72 true
optimization divisions, of which 67 were retained. Do not change source splits,
take audit labels, or search additional seeds/penalties to fit four validation
examples. The audit is `division-training-coverage-v1-result.json`.

Use the existing, hash-verified ZebraHub ZSNS004 TRAINING split: 64 transitions,
4,009 source nodes, 425 two-child lineage labels. Do not open ZebraHub validation
or audit shards. These are public tracking-derived labels, not a new manually
curated ground truth. The official imaging page identifies Ultrack as the source
of tracks. Competition permission/no-test-overlap was rechecked through the
organizer discussion. No claim that the public data has a specific redistribution
license; the final artifact/license packet remains a submission requirement.

Sources:
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330
- https://zebrahub.sf.czbiohub.org/imaging
- https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/

Frozen external manifest:
`b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf`.
Reuse unchanged Biohub v2 optimization/selection roles, original audit excluded,
manifest `39032899ecea16e87bfb53d45909f06993d39ff0cfbb0bbb16a7b875f9dcecce`.

## Domain alignment and examples

External source patches are centered temporally on t; external daughter patches
on t+1. Biohub relational patches at all three spatial centers are centered on
t. Use the common actual image pair t,t+1 at every spatial center, represented
as channels [t,t,t+1] to retain the frozen encoder's three-channel interface.
There is no invented image, unavailable t-1 daughter image, or t+2 look-ahead.
Both datasets use the existing one-micron 17-cubed grid, per-channel sample
standard deviation and +/-6 clipping, stored as float16 before context features.
Explicit frame-index regression tests must pass before creating the dataset.

Create one external positive for every true two-child row, retaining the lower
ID daughter first. Include all 425 positives, without excluding the one wider
than the legacy sister radius from TRAINING. For each parent with one or two
known children, include at most two non-child candidates within the unchanged
12-micron proposed-parent and 15-micron sister limits, ordered by descending
fixed biological-geometry score, then distance and ID. A non-child is a negative
only under these external lineage labels; do not call it manually curated.
No new randomness, confidence threshold, temporal resampling, or model-generated
negative labels. The deployment-eligible mask includes the same distance and
geometry>=3 conditions. No inference radius or quality gate is changed.

Image-derived context is calculated from the aligned parent image pair and
three anchors only. Existing image peak/half-max-volume descriptors and the
1,346-dimensional fixed encoder/geometry/morphology feature recipe are retained.
Parent velocity and its derived midpoint error are unavailable in the external
shards: set BOTH fields missing in BOTH domains, retaining the existing explicit
missing indicators, to avoid a trivial domain-identity feature.

## Training and qualification

For each reciprocal model, use its exact existing external-only backbone warm
start, frozen. Its head sees only the common external training split plus ONE
Biohub source embryo's optimization data. Give the two domains equal total
weight; within each domain retain existing class/eligibility-stratum balancing
and per-example quality weights. Fit normalization on optimization inputs with
equal domain mass (no labels used to fit normalization). Same clipping and
block scaling as frozen-head v1. Fit ONE logistic head, L2=0.01, analytic
L-BFGS-B, at most 300 iterations; no regularization sweep. Require convergence.
Selection only establishes the source threshold, never an alternative recipe.

Unchanged source gate: eligible AP>=0.55 and >=2 TPs above every source negative.
Freeze both source heads and thresholds before any opposite-embryo prediction.
Unchanged transfer gate: eligible AP>=0.55 and >=1 TP per embryo; >=3 pooled TP
and <=1 pooled FP at frozen source thresholds. A failure closes this recipe;
no automated extra weights, alternative source mixes, penalties or thresholds.
Historical project-level exposure is acknowledged. This is model-level embryo
exclusion, not pristine project-level validation. Patch passing still requires
actual public-baseline predicted candidates, division-positive complete movies,
patched official/worst-movie checks and offline two-GPU runtime before submission.

Use the already verified Antelume base runtime and warm starts in place. Stage
only external training shards/descriptors and small new code. One idle A10G,
sequential extraction, 2 CPU threads, 45% CUDA cap, no installation or RSNA
mutation. Actual-image/label small smoke and serialization replay first. Full
screen maximum 15 minutes; preserve completed feature banks and head files.
No Kaggle GPU use. Never stop the shared EC2 instance.
