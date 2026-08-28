# Multiscale contextual v4 post-v3 autolaunch

Date: 2026-08-28

## Outcome

The 46,386,607-parameter-per-fold multiscale contextual v4 pretraining lane is
preflighted and conditionally armed. It cannot launch until the contextual-v3
candidate has produced a verified competition submission receipt, so this
capacity experiment cannot consume quota ahead of or delay the first accepted
submission.

After that receipt exists, the controller may use all available Kaggle GPU quota
with zero reserve. It still requires at least 7.0 hours before launching the
24,000-second, exactly-two-T4 stage.

## Evidence repaired

The earlier staging report recorded notebook metadata hashes from before the
private ZSNS001 source slug collision was repaired. The current deterministic
artifacts are:

- Pretraining notebook:
  `c15f7e1d9080067bec78e7567b1aba8c95132b135854cedd56951954083ed4ec`
- Pretraining metadata:
  `d7edfb27e13d11c7cc3d9e85d3683417193e6acafc4ed3c0eb13362ef39504c6`
- Future reciprocal-transfer notebook:
  `fd749af69a332f06c012983dd8cb588e4363a7e3d3e9d07103ca6f544e14d71c`
- Future reciprocal-transfer metadata:
  `ec785539b7006808266cc2e81ee2017da5ab77dd1974d93dabb63195dbe36e83`

Both metadata files now use the collision-free private acceptance source
`indarkarhana/biohub-zsns001-contextual-gate-v1`.

## Preflight

The fresh append-only preflight verifies:

- private multiscale runtime dataset version 3 and its 39-file manifest;
- 64 ZSNS004 training and 16 disjoint ZSNS005 validation shards from dataset
  version 4;
- both strict-loaded 20,747,761-parameter v3 warm-start checkpoints;
- their passing untouched-ZSNS001 acceptance and terminal hashes;
- the exact 46,386,607-parameter v4 inventory and finite forward/backward tests;
- exactly two GPUs, T4 shape, internet off, and no competition input;
- no public predictions, public code copying, leaderboard selection, metric
  hacks, submission construction, or submission command.

The initial pre-amendment evidence remains preserved as
`artifacts/preflights/zebrahub-multiscale-contextual-pretrain-v1.json`
(`ff6f5edcb7082cc6e3abe4b817d2b384edf78bd5a43dd38b418203160aaecc16`)
rather than being overwritten.

Preflight report content SHA-256:
`de1f2fe396b7c5c4ebe11096b1602cbb247335a2c1b5ca862b6fa83bd491b381`.
Immutable report file SHA-256:
`1e36b0bd4f3db8a7fe128ab5ac146e7da1de43c7242d415d6c1f464bfca0c104`.

The six focused multiscale suites pass 22 tests. The controller-focused subset
adds 16 passing tests. No GPU was used during this work.

## Controller contract

`wait-launch-zebrahub-multiscale-contextual-pretrain.ps1`:

1. hash-verifies the notebook, metadata, experiment config, preflight builder,
   and immutable preflight report;
2. waits for the verified v3 submission terminal and matching receipt;
3. verifies the two accepted v3 source kernels are complete;
4. requires runtime dataset version 3 and shard dataset version 4;
5. waits for at least 7.0 GPU hours, with zero reserved hours;
6. pushes the private v4 notebook once and records its exact remote version;
7. never calls the competition submission API.

Controller SHA-256:
`bee24829f75e5fe0fac0a2e61963b266929d9149620ba233ed72c4c5e55b7465`.
