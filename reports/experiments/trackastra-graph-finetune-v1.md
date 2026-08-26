# trackastra-graph-finetune-v1

Status: staged for guarded GPU launch

## Hypothesis

A 27.5M-parameter Trackastra association transformer pretrained across diverse
3D Cell Tracking Challenge datasets, then fine-tuned on Biohub lineage graphs,
will generalize better than the public notebook's 8M-parameter two-frame linker.

## Why this is not the public 0.927 replica

- The public submission CSV and test graph outputs are absent from the runtime.
- The candidate uses Trackastra's six-layer encoder plus six-layer decoder,
  four-frame temporal context, rotary positional encoding, and 512 hidden units.
- Supervision is rebuilt from Biohub ground-truth graph edges.
- Biohub physical anisotropy scales Z by 4 relative to XY.
- Training simulates detector misses, false positives, coordinate jitter, dense
  hard negatives, and explicitly oversamples/upweights divisions.
- Dense frames use recursive source tiling with target halos so tokens are not
  silently dropped at Trackastra's fixed context limit.

Frozen raw prediction graphs from the public detector are used only to measure
the new linker's complete-movie held-out score on identical node coordinates.
They are not submitted and are not training labels.

## Resource envelope

- Declared GPU maximum: `4.00 h`
- Notebook watchdog: `14,100 s`
- Trainer wall budget: `13,200 s`, reserving at least `1,800 s` for validation
- Target: up to `6,000` sampled windows, stopping on time first
- Protected Kaggle GPU reserve: `8.00 h`
- Sequential execution; Internet and TPU disabled

## Validation and decision

Four complete held-out movies span both embryo prefixes. The run sweeps linker
thresholds once over frozen model probabilities, using adjusted edge Jaccard plus
`0.1 * division Jaccard`. The public leaderboard is not a selection signal.

Promotion requires a positive delta over clean proxy `0.929443`, plus no material
worst-movie regression. This run produces model and validation evidence only; it
does not run test inference or create a Kaggle competition submission.

## Provenance

- Trackastra repository: `weigertlab/trackastra`
- Pinned commit: `6a8ce94ee7c5a1f22c8eb77229ea5a0bc95a7b5b`
- Official `ctc` v0.3.0 model SHA-256:
  `24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f`
- License: BSD-3-Clause
- Private runtime dataset: `indarkarhana/biohub-trackastra-graph-runtime-v1`
