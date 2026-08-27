# ZebraHub contextual acceptance v1

Status: verified and unopened by any model.

## Purpose

ZSNS001 is a third embryo reserved for one-shot post-selection evaluation of
the project-authored contextual v3 appearance model. It is not available to
ZSNS004 optimization, ZSNS005 checkpoint selection, or ZSNS005 audit. The
inventory was fixed before v3 weights existed and may not redirect checkpoint,
hyperparameter, or submission selection.

This extra embryo is motivated by the primary Trackastra study's result that a
general model trained across diverse domains outperformed specialized models,
including out of domain. It is used here as a generalization test, not as a
source of public code or predictions:
<https://arxiv.org/abs/2405.15700>.

## Frozen public source

- Source: `ZSNS001`
- Role: `external_acceptance`
- Public inventory:
  <https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/>
- Physical metadata:
  <https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr/.zattrs>
- Organizer authorization:
  <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330>
- Level-0 spacing: `(1.24, 0.439, 0.439)` micrometres
- Level-1 spacing: `(2.48, 0.878, 0.878)` micrometres
- Image frames: `791`
- Track CSV bytes: `890,580,527`

ZSNS001 exposes track-level rows
`track_id,t,z,y,x,parent_track_id`, unlike ZSNS004/005's node-level IDs. The
project-owned converter maps a continuing target to its same `track_id` in the
previous frame and maps a newly appearing daughter to `parent_track_id`. A
synthetic continuation/division test pins this behavior.

## Immutable derived asset

- Local root:
  `.biohub/staging/biohub-zebrahub-contextual-acceptance-v1`
- Derived bytes: `8,496,478`
- Files: `41`
- Shards: `16`
- Selected timepoints:
  `120, 300, 480, 660, 121, 301, 481, 661, 122, 302, 482, 662, 123, 303, 483, 663`
- Source nodes: `1,023`
- Target nodes: `1,536`
- Candidate edges: `80,485`
- Positive edges: `1,150`
- Division sources: `127`
- Inventory SHA-256:
  `e32bc686e14222e43acb8d6247351e286eae8ed6fdb1f4ab5087e55fb0c79667`
- Manifest SHA-256:
  `cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0`
- Private Kaggle dataset:
  `indarkarhana/biohub-zebrahub-contextual-acceptance-v1`, version 1
- Remote re-download:
  `.biohub/cache/dataset-redownloads/biohub-zebrahub-contextual-acceptance-v1-version1`

Only normalized temporal 17-cubed patches, physical coordinates, candidate and
positive masks, division targets, and label-free transition/candidate context
are included. The raw CSV and OME-Zarr chunks remain below ignored cache paths
and are absent from the derived asset.

The independent verifier checked every shard array, balanced sampling rule,
hard negative, lineage mask, byte count, content hash, manifest, generator,
and source file. Its terminal state is `verified_unopened` and records:

- `model_predictions_read: false`
- `competition_test_data_read: false`
- `public_competition_predictions_read: false`
- `leaderboard_used: false`
- `submission_created: false`

The private version-1 publication was re-downloaded through the Kaggle CLI and
the downloaded copy reproduced the exact manifest, inventory, counts, and all
16 shard checks. The first relative-path upload attempt failed locally before
dataset creation; the successful upload used an absolute path and Kaggle's
zipped-directory mode. No alternate dataset version exists.

## Future one-shot gate

The asset stays unopened until both v3 folds pass their disjoint ZSNS005
selection and audit gates and their checkpoint hashes are frozen. Each final
fold will then be compared once with its exact seeded initialization using the
same composite, top-1, MRR, and division-top-2 metrics. The third-embryo audit
must show at least `0.01` composite gain, strictly improve top-1 and MRR, and
must not reduce division-top-2 recall. Failure rejects v3; no result may be used
to tune or choose another checkpoint.

No Kaggle GPU and no competition submission were used to build this asset.
