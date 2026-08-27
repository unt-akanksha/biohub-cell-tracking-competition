# ZebraHub contextual pretraining v1 launch

Status: guarded two-GPU run started.

## Scientific prerequisite

The parent `temporal-patch-dual-fold-v1-inventory-repair` completed and passed
strict checkpoint verification before this launch. Both independent cosine
folds improved real association ranking and synthetic division ranking. It is
retained as a control and is not authorized for submission. Its result is
`reports/experiments/temporal-patch-dual-fold-v1-inventory-repair-result.json`.

## Guarded launch

- Run ID: `zebrahub-contextual-pretrain-v1`
- Kernel: `indarkarhana/biohub-zebrahub-contextual-pretrain-v1`
- Started event: `evt-8d6fd529cd934e699fc20e5803df791e`
- Authorization: `auth-eea4ebbfd1ca4275bcf5c04f974302e2`
- Authorization SHA-256:
  `6fc2d6fadc4233aca4e04e042508a200e7bfad3daf7767581d948e45e3ba943e`
- Kernel source SHA-256:
  `a5fc12dc663ab0daaea1fabf56d536423c0fbb73336f9553bebf322d2bd094c7`
- Preflight SHA-256:
  `d5e8f1b9b1d2a10dda851d3f3559f7b2325f46302ae69d95d87c22c4b097859f`
- Declared maximum: `6.67` hours
- Notebook watchdog: `24,000` seconds
- Worker/orchestrator ceilings: `21,600 / 22,800` seconds
- Live remaining quota: `17.45` hours
- Projected remaining after full declared ceiling: `10.78` hours
- Required reserve: `8.00` hours
- Active GPU kernels at authorization: none
- Exact visible GPUs required at runtime: two T4 GPUs

The launch followed register, immutable preflight, live authorize, and
single-use execute. Direct unguarded kernel push was not used.

## Frozen inputs and gates

- Private runtime: version 8, manifest
  `85cfd63f75340502e3c810d71a8006fd15342dbc263f6ae45b0c376cf9b1ff7b`
- Private balanced shards: version 4, manifest
  `b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf`
- Optimization source: 64 ZSNS004 shards only
- Checkpoint selection: ZSNS005 `t0096-0099` and `t0376-0379`
- One-shot in-run audit: ZSNS005 `t0236-0239` and `t0516-0519`
- Independent seeds: `51004` and `61007`
- Parameters per fold: `20,747,761`
- Pair objective: outgoing all-positive child ranking plus `0.35`-weighted
  eligible incoming-parent ranking
- Augmentation: fixed-scale microscopy rotations/flips, gain, and mild noise
- Internet: disabled
- Competition source: absent
- Submission command: absent

Each fold must improve composite by at least `0.01`, strictly improve top-1 and
MRR, preserve division top-2 recall, and retain the exact inventory on both the
selection and audit partitions. Failure in either fold rejects the run.

The separately frozen private ZSNS001 acceptance set remains unopened until
both ZSNS005 gates pass and checkpoint hashes are final. Public notebook code,
public predictions, leaderboard selection, metric hacks, and competition
submission are excluded.
