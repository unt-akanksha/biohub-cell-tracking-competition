# Temporal patch v1 inventory-repair launch

Date: 2026-08-27

Status: launched for reciprocal training; no inference, candidate, or
competition submission was created.

The repair binds the exact attached competition inventory (`44b6: 71`,
`6bba: 128`) and the already intended up-to-96 partition policy. After two
opened-movie exclusions plus 12 validation and 12 calibration reservations per
prefix, the effective real training counts are 96 for `target_44b6` and 45 for
`target_6bba`. Architecture, losses, 30,000 steps, real replay probability,
seeds 45427/47627, and all clean split boundaries are unchanged.

Local gates before launch:

- inventory-repair preflight SHA-256:
  `600a9859a863ce1f064b1a926119b3da516f16bf0d1706d09451b9c75e610870`;
- focused repair/runtime suite: 51 passed;
- full repository suite: 552 passed, 2 expected Windows symlink skips;
- private v1 runtime manifest:
  `131a35a3e72d4a0ba4e81c8af075613d1a73dc8e465d4a80282b558cd844af0d`;
- fresh private Kaggle runtime version contained 31 uploaded entries at
  2026-08-27 09:17:42â€“09:17:43 UTC;
- independently extracted archive verification: 30 files, 440,745 bytes,
  exactly two GPUs required, no submission command.

The one-use live guard observed no active kernels and 21.13 Kaggle GPU-hours
remaining. With the immutable 11-hour run ceiling, it projected 10.13 hours
remaining, above the protected 8-hour reserve. Authorization
`auth-c3170c84040245178c0701944506a571` was consumed once; the ledger start
event is `evt-7065e151e5ff44e4aff045e853101382`.

The run ID is `temporal-patch-dual-fold-v1-inventory-repair`, the Kaggle kernel
is `indarkarhana/biohub-temporal-patch-dual-fold-v1`, GPU shape is T4 x2, and
internet/TPU are disabled. The running kernel will not be repeatedly polled;
terminal evidence is the next operational status gate.
