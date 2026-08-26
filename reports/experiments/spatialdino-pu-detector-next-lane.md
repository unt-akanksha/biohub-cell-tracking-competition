# SpatialDINO PU detector: independent heavy-model lane

Date: 2026-08-26

Status: staged and preflight-passed; intentionally not launched while
`spotiflow-pu-adaptation-v1` is active.

## Why this is not a public replica

The candidate produces a new learned checkpoint for a 29,521,225-parameter
architecture that does not exist in the public Biohub notebooks. It combines:

- the official microscopy-pretrained SpatialDINO ViT-S/8 encoder;
- a trainable raw-image 3D convolutional pyramid preserving fine structure;
- a UNETR-like decoder fusing four transformer depths; and
- a full-resolution heatmap head with sub-voxel soft-centroid inference.

The two public TemporalUNet seeds are frozen and used only to identify
high-confidence consensus positives and unknown support. Public prediction
graphs are neither copied nor used as detector outputs. Organizer annotations
are forced positives. Teacher disagreements have zero loss and safe background
has weight 0.01.

## Resource-aware training design

- 8,019,913 decoder parameters train during warm-up.
- The last four SpatialDINO blocks then open, for 15,121,609 trainable
  parameters while earlier encoder blocks remain frozen.
- EMA weights with decay 0.995 are saved for clean validation.
- One deterministic frame pair from every non-validation movie supplies broad
  embryo coverage; both pair frames are visited across cycles.
- Teacher targets and raw frame pairs are cached on CPU. This converts repeated
  teacher inference after the first coverage cycle into cheap cache hits.
- Training is batch-one; clean inference is batch-four with Y/X flip TTA.
- A 5,200-second trainer guard and 1,300-second evaluator guard fit inside the
  6,900-second notebook watchdog.

## Clean evidence gates

All twelve fixed validation movies are excluded from training and target reads.
Eight selection movies must reach pooled annotated recall 0.80 and worst-movie
recall 0.65 before the four acceptance movies can be opened. Promotion then
requires at least the public baseline pooled recall (0.96902842596521), both
embryo-prefix recalls within 0.01 of their baselines, and no movie delta below
-0.01. Thresholds are frozen from image responses and organizer estimated node
counts before ground-truth scoring. The leaderboard is not used for selection.

## Verification

The preflight passed all eight checks. The focused suite reports 32 passing
tests, including strict SpatialDINO loading, intermediate-feature equivalence,
full-resolution model geometry, trainable-phase isolation, EMA behavior,
positive-unlabeled masking, sub-voxel inference, clean-result provenance, and
the future two-GPU submission policy.

Preflight report SHA-256:
`b950d3c2ffc086a540430dc8ac2145f43353a36996d47e9dca9fdb3293a2f6fc`

## Launch rule

Do not run concurrently with the active Spotiflow GPU experiment. Once that run
is terminal and reconciled, this lane may be uploaded and launched only if the
quota projection still preserves at least eight Kaggle GPU hours. It remains a
training/validation notebook and cannot create or submit a competition file.
Any later submission notebook must use two GPUs and shard movies across them.
