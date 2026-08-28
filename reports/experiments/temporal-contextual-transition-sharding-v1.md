# Temporal contextual transition sharding v1

Date: 2026-08-28

## Outcome

The final contextual-v3 candidate now partitions independent consecutive-frame
transitions across exactly two Kaggle T4 GPUs. This removes the severe whole-movie
load imbalance that made the earlier final-inference design vulnerable to the
12-hour notebook limit, without changing the frozen model, calibration, association
configuration, nodes, or transition inventory.

## Real test-inventory audit

The hash-bound clean base submission contains four 100-frame movies. The conservative
contextual pair-fusion workload estimates are:

- `44b6_0113de3b`: 7,303,666,688
- `44b6_0b24845f`: 4,633,891,840
- `6bba_05b6850b`: 537,174,016
- `6bba_05db0fb1`: 52,856,700,928

The dominant `6bba_05db0fb1` movie represents 80.9055% of the estimated work.
Whole-movie LPT therefore assigned 52,856,700,928 units to one GPU and
12,474,732,544 to the other, a 4.2371x load ratio.

The transition planner splits only the dominant movie at a consecutive-frame
boundary:

- GPU 0: `6bba_05db0fb1@0-32` plus all three smaller movies,
  estimated load 32,641,068,032.
- GPU 1: `6bba_05db0fb1@33-98`, estimated load 32,709,755,904.

The resulting ratio is 1.0021x. The real-inventory work-plan SHA-256 is
`84dd106082352ac9f4751c6be455c65485c3b07e6bd3fe87b9d9ac63eb8bd960`.

## Correctness gates

- Every one of the 396 `(movie, transition_start)` pairs is assigned exactly once.
- Each shard receives complete source and target frames for every assigned
  transition.
- Trackastra scoring, contextual pair scoring, reciprocal blending, and hybrid
  association all operate independently per consecutive-frame transition.
- Recombined edges are rejected on duplicate units, duplicate edges, unknown
  units, out-of-block sources, missing transition coverage, invalid lineage
  degree, changed nodes, or an exact public-base replica.
- The final report must contain the transition work-plan hash and partition kind
  before `submission.csv` is exposed at the notebook root.

Focused inference, runtime, kernel, launch, and submission tests: 32 passed.

## Runtime provenance

The final-only private dataset is
`indarkarhana/biohub-temporal-contextual-final-runtime-v1`, version 2. Version 1
is deliberately excluded because Kaggle CLI's default directory mode skipped the
nested Trackastra source. Version 2 was uploaded with zipped directory preservation,
redownloaded, and verified:

- Parent transfer-runtime-v1 version-4 manifest:
  `cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d`
- Final runtime manifest:
  `e69a20f10f56108818a6bf0715fe071e868fd176d1720cc2ffc04f2a645b41ff`
- Final inference source:
  `575f375a261b34cdda67709c4de46a6b64772ec347089e606e7c7dedae4716f7`

Only `dual_fold_appearance_submission.py` differs from the frozen parent runtime.
No competition submission command is present in the runtime or candidate notebook.
