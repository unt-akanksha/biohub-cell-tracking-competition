# Temporal patch v1 exact-inventory failure

Date: 2026-08-27

Status: infrastructure/data-contract failure before model execution. No
submission was created.

The mount-aware retry passed both runtime verifiers. It then failed after
510.6 seconds because the notebook required at least 120 complete image/GEFF
pairs for each embryo prefix. The exact attached competition inventory is 71
`44b6` movies and 128 `6bba` movies. The failure occurred before the production
dense probe, optimizer training, appearance checkpoint, calibration, processed
acceptance, candidate creation, or submission.

The trainer's predeclared partition already means “up to 96” real source
movies. After excluding two opened movies and reserving 12 validation plus 12
calibration movies from each prefix, the effective reciprocal training counts
are therefore 96 for `target_44b6` and 45 for `target_6bba`. The repair binds
those exact counts in the trainer and independent downloaded-output verifier;
the model, losses, seeds, 30,000 steps, 20% real replay, and reserved movies are
unchanged.

Evidence:

- launcher terminal SHA-256:
  `ba97616e94b67503c55741a9e51e337bad6773504c08b6b0cba33c232e4dd249`;
- kernel log SHA-256:
  `a5e86dd9e6a15a8b04c33b3a75dcb8c3625b3c56f90883467a685e01eca335d5`;
- charged GPU time: 0.15 hours;
- remaining Kaggle GPU quota: 21.13 hours;
- ledger failure event: `evt-8993f4cfcdb94950a7be380edeef72e4`.

Any repair launch still requires a fresh live guard proving at least eight
hours remain after the immutable 11-hour ceiling.
