# Image-derived division context v1

September 13, 2026. New learned-model experiment, not a relaxation of any
rejected graph-context or public D4/repair gate.

Source inspection establishes that the prior graph-context training archive
uses sparse ground-truth node coordinates for neighborhoods, while deployed
context uses dense predicted nodes. Its edge-array exclusion does not make
those coordinates independent detections. Quantify the covariate shift on
optimization inputs; do not claim causality from input counts alone.

Replace annotation-neighbor tokens with image-derived peaks and half-max
connected-component volumes on the existing parent-centered image triplet.
No surrounding GT or predicted graph is accepted by the new feature function.
It receives only images plus the three candidate anchors, which are legitimate
supervised training positions and predicted positions at inference.

The grid is already one micron per voxel, 17 cubed, t-1/t/t+1. Use fixed
Gaussian sigma 1 micron, 3-voxel local maxima, 3-micron NMS, and at most eight
peaks per frame. Object size is the connected half-max component in a radius-4
box; contrast is relative to per-frame background/p99. This yields up to 24
image tokens plus three anchors in the existing 43-by-8 architecture. Remaining
slots are always masked. No source labels, model scores, or leaderboard values
choose these constants. They are frozen before real feature generation.

The recent public nucleus-size EDA motivates separating size from brightness,
but these preprocessed, clipped and standardized patches cannot measure raw
fluorescence or calibrated nuclear volumes. Do not repeat that stronger claim.

Use the existing 74,732,308-parameter image/transformer architecture with exact
46,386,607-parameter **external-only** warm starts. Do not initialize from any
Biohub-trained graph-context model. Keep every older model/result immutable.

Prospective two-fold pilot: for each held-out embryo, optimization and checkpoint
selection may use only the other embryo's frozen v3 optimization and selection
roles respectively. Do not open v3 audit shards. Held-out-embryo predictions may
be generated only after both models and their source-selection thresholds are
frozen. These rows have historical exposure from other experiments; report
model-level embryo exclusion honestly, not a pristine project-wide sealed audit.

First build and verify image features, then run a 20-step actual-model GPU smoke.
The pilot is two sequential 2,000-step models, seed 20260913/20260914, batch 10,
AMP, AdamW, LR 6e-5/head and 0.15 multiplier/backbone, weight decay 2e-4, EMA
0.995, clip 2, cosine minimum 3e-7, selection every 250 steps. Preserve optimizer,
EMA, best checkpoint and RNG on disk. Antelume only, initially idle GPU, two CPU
threads, 70% CUDA allocator ceiling, 40-minute pilot watchdog. Do not launch the
pilot unless measured smoke throughput predicts completion with margin. No
automatic larger extension or extra seeds, and no shared-project deletion.

Each source-selected member must achieve eligible AP >=0.55 and at least two
true positives at zero false positives using its source-only selected threshold.
Only then evaluate the model on the other embryo: require AP >=0.55, at least
one correctly recovered eligible division per embryo, at least three combined
true recoveries and at most one combined false recovery using the unchanged
source thresholds. No target-embryo threshold search, seed/model selection, or
rerouting. A pilot pass still requires actual predicted-candidate and complete-
movie official-score evaluation with division-positive coverage, worst-movie
checks, offline/two-GPU runtime proof, and non-replica/provenance checks before
submission. Dataset or functionality evidence alone never authorizes submission.
