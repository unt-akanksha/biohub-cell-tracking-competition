# Spotiflow detector lane for Biohub

## Decision

Use Spotiflow as a transfer-learning detector candidate, but do not fine-tune
it with a conventional fully supervised background loss. Biohub labels are
positive-unlabeled: across the 199 training fields in the locked manifest,
there are 133,318 ground-truth nodes versus 4,725,117 organizer-estimated total
nodes (2.82% labeled). Treating every unlabeled cell as background would train
the detector to suppress real cells.

The planned objective therefore combines:

1. high-confidence pseudo-positive peaks from the existing two-seed detector;
2. all Biohub ground-truth nodes as forced positive anchors;
3. a very low-weight unlabeled/background term; and
4. complete-movie validation with no test-label or metric feedback.

## Verified upstream assets

- Official repository: `weigertlab/spotiflow`, commit
  `b4f464552afc3424944a9c0fa60ac024e4013f1f` (BSD-3-Clause).
- Official release: Spotiflow model bundle `0.6.0`.
- `synth_3d.zip` MD5: `a031f1284590886fbae37dc583c0270d` (verified).
- `synth_3d/best.pt`: 35,489,892 trainable/non-buffer parameters and
  35,497,921 total state values, 142,064,642 bytes. The model is a four-level
  3D U-Net with 32 initial feature maps,
  stereographic-flow localization, and a `(2, 2, 2)` output grid.
- Pretraining used `(32, 128, 128)` crops, AdamW at `3e-4`, and 200 epochs.

Biohub raw voxels are `(1.625, 0.40625, 0.40625)` microns. The established
`(1, 4, 4)` spatial read produces isotropic `(1.625, 1.625, 1.625)` micron
voxels and `64 x 64 x 64` frames, so the pretrained 3D architecture is shape
compatible without inventing an anisotropic convolution convention.

## Lower-cost calibration experiment

The current public detector applies threshold `0.96875` to every movie, while
clean held-out final node counts range from 0.752x to 1.275x the organizer
estimate. `research/density_calibration.py` selects a movie-specific threshold
from uniformly sampled image-frame responses. It is deliberately asymmetric:
it may raise the threshold for a projected over-count, but never lowers it.
This avoids adding unvalidated low-confidence nodes merely because under-counts
receive a favorable count adjustment.

Promotion requires a complete held-out graph gain, stable node recall, and no
regression in division Jaccard. A detector-only or count-only proxy is not a
promotion gate.

## Dense pretraining source

The 2026-08-26 public audit identified a CC0 physical synthetic source with
1,539 fully labeled static volumes. Its static coordinates are suitable after
the same XY stride/downscale used by Biohub; its temporal coordinates require a
documented Y/X divide-by-four repair. If pretrained Spotiflow is viable, this is
the preferred dense-supervision stage before positive-unlabeled real-data
adaptation. See `research/SYNTHETIC_PRETRAIN_AUDIT.md`.

## Implemented candidate (2026-08-26)

`train_synthetic_detector.py` now implements the dense-supervision stage rather
than merely proposing it. It warm-starts the selected official 35,489,892
parameter checkpoint and trains on 1,200 deterministic corrected static
volumes, with 128 disjoint synthetic validation volumes. The first guarded run
uses 3,072 optimizer batches (four epochs of 768 replacement samples), batch
size one, `32x64x64` crops, AdamW at `3e-5`, and a two-hour wall envelope.

The static source remains attached as a Kaggle kernel output and is streamed.
Metadata-aware lazy array proxies prevent Spotiflow's shape-validation passes
from decompressing the full 13 GB static collection several times before the
first optimizer step. The run writes a source/hash manifest, incremental
metrics, best/last checkpoints, and terminal evidence, and has no submission
code. It is a new learned candidate, not a repackaged public prediction.
