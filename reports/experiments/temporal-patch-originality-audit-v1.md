# Temporal patch candidate originality audit v1

Date: 2026-08-27

Decision: the staged candidate is not an exact copy of a public Biohub
notebook or public prediction artifact.

## Provenance

- Association backbone: upstream Trackastra CTC v0.3.0 initialization, source
  commit `6a8ce94ee7c5a1f22c8eb77229ea5a0bc95a7b5b`, BSD-3-Clause. Two reciprocal
  27,456,880-parameter copies are fine-tuned independently on corrected public
  synthetic graphs and disjoint opposite-embryo real folds.
- Appearance backbone: project-authored 19,221,954-parameter physical-scale 3D
  residual encoder per fold. It starts without external pretrained weights and
  learns aligned `t-1,t,t+1` cell patches, all-daughter supervised contrastive
  links, and division evidence.
- Graph control: the clean public graph may be used as the frozen base node
  inventory, but no public prediction file is packaged into this runtime.
- Public implementations: no Biohub public-notebook code or weights are copied.
  The audited `arnav170/biohub-mtl8` notebook was explicitly rejected because
  its own receipt admits leaderboard-guided configuration and public-kernel
  derivation.

## Mechanical checks

The independently extracted portable runtime contains 26 hash-bound files,
requires exactly two GPUs, and contains no competition submit command. A scan
found none of the audited public-kernel owner or method identifiers
(`arnav170`, `pilkwang`, `yusuketogashi`, `raykkretzsch`,
`harmonic_probability`, or `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT`). The only
`public leaderboard` occurrence is a negative policy declaration that the
leaderboard is unused.

Both candidate builders fail if their output has no changed edges; the stronger
appearance builder additionally rejects a byte-identical base CSV. Therefore a
public-base replica cannot become accepted evidence or a local final candidate.

## Bound package

- Archive:
  `.biohub/staging/biohub-temporal-patch-runtime-v1-heavy-temporal3-framecache-strictingest-ema-ensemble-daughterloss-runtimeguard-t4x2-allreaped-20260827.zip`
- SHA-256:
  `39de54db40c9620591519d0c9bd3baaa89ebe314c2ef9a8045f7e43444e13230`
- Manifest SHA-256:
  `8d39fcedb424498e6e6dd07bc224ab6a1748c74c91959ee620906233a0ad066c`
- Competition submission performed: false

The candidate remains conditional on clean fold improvement, one-shot pinned
processed acceptance, and separate user authorization before any Kaggle
competition submission.
