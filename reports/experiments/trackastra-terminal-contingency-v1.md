# Trackastra dual-fold terminal contingency v1

Date: 2026-08-27

This is a fail-closed decision table for the single scheduled terminal check of
`indarkarhana/biohub-trackastra-dual-fold-synthetic-v1`. It authorizes no
competition submission and no additional Kaggle GPU run.

## Terminal handling

1. Query kernel status once in the expected completion window.
2. If terminal, download output once into a new empty cache directory.
3. Locate the directory containing `training_terminal.json` without modifying
   the output.
4. Run the hash-bound `verify_trackastra_output.py` from the final portable
   runtime.
5. Query accelerator quota once after the terminal result is secured.

## Accepted

Proceed to the cloud appearance handoff only if the verifier reports both
folds improved, exactly two GPUs, 27,456,880 parameters per fold, corrected
native synthetic geometry, exact split inventories, matching model/config
hashes, and no competition artifact. Stop Kaggle experimentation at the
expected roughly 20.45 GPU-hours remaining.

## Rejected or failed

- Missing/invalid aggregate terminal: preserve logs and classify the run as an
  execution failure; do not infer model quality.
- One or both `best_step == 0`, insufficient real gain, or excess synthetic
  regression: reject the models scientifically; do not materialize processed
  candidates and do not consume the one-shot processed gate.
- Hash, split, geometry, or worker/aggregate mismatch: quarantine the output as
  unverifiable even if metrics look strong.
- CSV, ZIP, or submission-named output: quarantine as a policy violation.

No failure path triggers another Kaggle GPU launch because the reserve boundary
has been reached. Any retry moves to the user-provided cloud instance, uses two
isolated GPU workers, retains unopened-fold selection, and must produce fresh
hash-bound terminal evidence before appearance training can begin.

## Submission boundary

Neither an accepted nor rejected training result authorizes a Kaggle
competition submission. Final submission remains conditional on clean
appearance calibration, one-shot pinned processed acceptance, a non-replica
candidate, the two-GPU T4 x2 runtime contract, and explicit user authorization.
