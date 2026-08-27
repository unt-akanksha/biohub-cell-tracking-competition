# Temporal contextual calibration v3 staging

Status: local two-GPU evidence kernel staged, verified, not pushed, not
launched, and not submitted.

This is the clean calibration stage for the project-authored contextual v3
candidate. It is not a public-notebook replica. The stage only becomes eligible
after both contextual competition-transfer folds pass their preregistered real
gain and synthetic-retention gates.

## Frozen comparison

- Contextual source: `temporal-contextual-pair-fusion-v3`
- Exact control: `trackastra-dual-fold-synthetic-v1`
- GPUs: exactly `2`, one reciprocal embryo fold per GPU
- Reserved movies: `12` per fold, fixed before transfer training
- Ensemble modes: `target_only`, `reciprocal_mean`
- Appearance weights: `0`, `0.05`, `0.10`, `0.20`, `0.35`
- Division weights: `0`, `0.05`, `0.10`, `0.20`
- Exact control point: target-only with both weights equal to zero
- Minimum pooled composite gain: `0.001`
- Maximum regression on any reserved movie: `0.002`
- Both folds must improve before promotion
- Processed acceptance labels: unopened
- Public predictions and leaderboard selection: forbidden
- Submission construction: absent

The zero point evaluates pure Trackastra with no contextual appearance or
division contribution. If no grid point passes the pooled and worst-movie
gates, the calibrator fails closed to this control and records the family as
not improved.

## Frozen local package

- Runtime dataset: `indarkarhana/biohub-temporal-contextual-transfer-runtime-v1`
- Only admissible version: `3`
- Runtime manifest SHA-256:
  `193478079a0d3f83c1307416c60ed5ad7c74a840fafc5e046ef7f30e2db4b3c1`
- Kernel: `indarkarhana/biohub-temporal-contextual-calibration-v3`
- Notebook SHA-256:
  `525126fc8008aba78fcefea6093ef9a32a610e7e88fab2f03c1aa0576d11560a`
- Kernel metadata SHA-256:
  `46f0b73396f50d1e3bebb40ce2f811b20d791347dbf51b0b512e2e71e492b955`
- Notebook watchdog: `21,600` seconds
- Worker limit: `18,000` seconds
- Orchestrator hard stop: `19,800` seconds
- Internet: disabled

The kernel strict-verifies the contextual and Trackastra checkpoints before
calibration. It contains no processed-acceptance read, competition artifact
builder, or Kaggle submission command. Execution is reserved for the later
two-GPU cloud stage so the active Kaggle pretraining run remains within the
eight-hour quota reserve policy.
