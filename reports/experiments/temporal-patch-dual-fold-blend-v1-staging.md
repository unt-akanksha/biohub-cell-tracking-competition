# Temporal patch dual-fold blend v1 staging

Date: 2026-08-27

Status: registered and preflight-passed, not launched. The reciprocal v1
training kernel remains the only GPU experiment in flight. This calibration
kernel cannot build or submit a competition artifact.

## Scientific boundary

- Parent: `temporal-patch-dual-fold-v1`, a project-authored 19,221,954-parameter
  temporal 3D appearance encoder per fold; it is not a public-notebook replica.
- Geometry control: the coherent, hash-bound predeclared Trackastra model.
- Selection data: 12 deterministic movies reserved before training for each
  embryo prefix. The four processed-acceptance movies remain unopened.
- Search: one frozen grid over target-only versus reciprocal-mean evidence,
  appearance weights `0, 0.05, 0.10, 0.20, 0.35`, and division weights
  `0, 0.05, 0.10, 0.20`. Exact zero is the control.
- Gate: pooled clean-link gain at least `0.001`, worst-movie regression no worse
  than `-0.002`, and both reciprocal folds must improve.

## New fail-closed boundary

Calibration cannot start unless an independent verifier proves:

- aggregate status completed, exactly two training GPUs, and both folds trained;
- each worker terminal equals its aggregate fold row exactly;
- both checkpoint hashes match and both state dictionaries strict-load;
- family, architecture, loss, EMA, and 30,000-step ceiling metadata are exact;
- held-out real top-1 is at least `0.70` and held-out synthetic top-1 at least
  `0.85` for each selected checkpoint;
- the 1,900/128 synthetic partitions agree across folds;
- each real prefix's 96 training, 12 checkpoint-validation, and 12 calibration
  movies are globally disjoint and exclude opened acceptance stems;
- no CSV, ZIP, submission path, public predictions, or leaderboard selection
  appears in training output.

Processed acceptance independently repeats worker/aggregate and model/calibration
binding checks, so later mutation of the downloaded source also fails closed.

## Execution and quota

- Kernel: `indarkarhana/biohub-temporal-patch-dual-fold-blend-v1`
- Accelerator: exactly two `NvidiaTeslaT4` GPUs; TPU and internet disabled.
- Worker/orchestrator/notebook ceilings: `18,000 / 19,800 / 21,600` seconds.
- Declared quota ceiling: `6.00` hours.
- Launch condition: parent training passes strict verification and a fresh live
  Kaggle guard still projects at least `8.00` hours remaining. Otherwise this
  stage waits for the user-provided cloud GPUs.
- Competition submission: not authorized and no submit command is present.

Final preflight report:
`artifacts/preflights/temporal-patch-dual-fold-blend-v1-coherence2.json`,
SHA-256
`0c8e1ea41d6f43619a087617a4f1e4fe123ddcc9b5a1ff28b869664aa499c3db`.
The earlier immutable report is retained as historical evidence rather than
overwritten.
