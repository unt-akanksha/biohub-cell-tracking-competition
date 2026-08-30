# Biohub Experiment Log

## 2026-08-30 — overnight heavy-model schedule

Target: a clean, non-replica candidate aimed at 0.945, with no leaderboard or
metric-hack selection.

### Stage A: deep seed sweep

- Antelume A10G, one GPU, deployed before this log entry.
- Eight independent seeds crossed with two initial models: 16 models total.
- 50,000 training steps per model (800,000 model-steps total).
- Selection and sealed-audit evidence are produced before any competition
  candidate can be assembled.
- Last projected completion window was approximately 06:00–07:00 America/Chicago.
  This is a projection, not an acceptance result.

### Stage B: relational division sweep

- Antelume A10G follow-on job, resource-gated behind Stage A and an idle-GPU
  check so the jobs cannot contend.
- Four new seeds crossed with two distinct initial backbones: 8 independently
  trained 48,313,050-parameter models.
- 15,000 full-network steps per model; each step encodes the parent, retained
  daughter, and proposed daughter, for 120,000 model-steps and 360,000 relational
  patch encodings before validation work.
- Selection gate: AP at least 0.55, embryo AP at least 0.40, and two true
  positives before the first false positive.
- Ensemble membership is frozen on selection. The sealed audit cannot search
  member subsets, and every deployed member must pass independently.
- The accepted relational voter must agree on the identical top candidate with
  the independent 132-feature temporal morphology voter. At most one additive
  edge is allowed per movie; reassignments and node/coordinate edits are banned.

### Candidate and submission boundary

- A development-positive result packages a private, hash-bound runtime but does
  not authorize submission.
- The Kaggle notebook is private, offline, competition-attached, and requires
  exactly two T4 GPUs. Accepted relational models are partitioned over both GPUs
  when at least two members are admitted.
- External promotion requires proxy gain at least 0.005, improved division true
  positives and division Jaccard, adjusted-edge regression no worse than 0.001,
  no known-public submission hash, and exact additive-graph invariants.
- The submitter has an exactly-once receipt and a five-submissions-per-day cap.

### Automation state

- Relational extractor: complete, 2,274 shards / 3,013 candidates, archive SHA-256
  `66a822bce0c60d06f6a2b60ada313f0d4d55062de1f84fb60bded4ae456266c2`.
- Relational deploy controller: credential-waiting; the AWS STS session was
  expired at 02:27 America/Chicago. The active Stage A remote job is unaffected.
- Relational harvest, development evaluation, runtime packaging, dual-T4 launch,
  promotion, and submission are event-driven and already chained.
- Dual-T4 launch controller PID after verified handoff fix: 44240.

No public leaderboard result is used as training, member selection, ensemble
selection, or acceptance evidence in this schedule.

## 2026-08-30 — graph-context model family

Motivation: additional capacity without the correct inductive bias was not
sufficient. The rejected Kinetics-pretrained Swin3D-B ranker had 87,640,009
parameters but only 0.4901 sealed-audit AP and zero recall at zero false
positives. A domain-trained patch voter reached 0.6325 AP on the final frozen
probe, but its selection-frozen absolute threshold selected no events. The next
experiment therefore adopts Trackastra's applicable principle: score a candidate
with the surrounding spatiotemporal detection set, not only isolated crops.

Primary design reference:
https://arxiv.org/abs/2405.15700

### Leakage-safe data enrichment

- Input: the exact relational v3 archive, SHA-256
  `66a822bce0c60d06f6a2b60ada313f0d4d55062de1f84fb60bded4ae456266c2`.
- GEFF access is restricted to five node arrays: ID, time, z, y, and x. No edge
  array is opened, so audit division labels cannot enter the context features.
- Each candidate has 43 tokens: parent, two symmetrically typed daughters, and
  up to eight nearest detections at each of five relative timepoints.
- Exact daughter swapping and arbitrary context-token permutation leave model
  output unchanged.
- Output reproduces all 3,013 examples, 134 positives, 2,879 negatives, 55
  inference-eligible positives, and 165 inference-eligible negatives across 146
  movies and 2,274 shards.
- Valid context tokens: 48,307.
- Manifest SHA-256:
  `2e5c4c46b11ff1480194c29e88c0e16b0ebf2f0fd65b2fc29548d26fee599bd9`.
- Deterministic archive SHA-256:
  `efa3b5af80c75b2bc091ecda1e86064660dcfffe197e2a4950a12981248b0c3e`.

### Model and run schedule

- Model: the 46,386,607-parameter microscopy backbone plus an eight-layer,
  512-dimensional, eight-head detection-set transformer and rank head.
- Total parameters per model: 74,732,308; graph-context/rank parameters:
  28,345,701.
- Schedule: four seeds crossed with two distinct initial backbones, 8 models,
  20,000 full-network steps each (160,000 model-steps).
- Execution is chained after the 48.3M relational sweep terminal and an idle-A10G
  check. It cannot contend with the earlier sweep.
- Selection and audit gates match the relational experiment. Ensemble members
  are frozen before audit, every deployment member must pass independently, and
  deployment uses ranks rather than an absolute threshold.
- Deployment controller PID after development-scorer wiring: 30028. It waits on
  the relational deployment event and refreshed AWS credentials.
- Harvest controller PID: 43984. It opens one authenticated transfer only after
  deployment and verifies the complete hash-bound archive before extraction.
- Frozen development inventory: 225 rows, 9 inference-eligible candidates, 3
  positives, and 9,666 valid context tokens; SHA-256
  `c8883e77abcf76c5a837c5fd0b21afdfefb2e51cb8e550e3a81f69d70562f311`.
- Development/runtime controller PID after tested runtime-packaging refresh:
  48016.

### Candidate chain

- A remote exit code of zero now requires the precommitted deployment policy to
  pass sealed audit. An independently strong member alone is insufficient.
- Frozen development acceptance requires the graph-context and independent
  morphology rankings to select exactly the same three events, all three true
  positives and zero false positives, with no regression from the pinned EMA
  development baseline.
- A positive development terminal packages a private, hash-bound runtime. It
  still has no submission authority.
- The candidate notebook requires exactly two T4 GPUs, partitions independently
  strong 74.7M members across both devices, averages member ranks without an
  absolute threshold, and permits at most one additive edge per movie after
  exact top-rank agreement with morphology.
- Graph-context launch controller PID: 19764. It waits for the scientific
  runtime event, uploads the private dataset, launches the exact private kernel
  version, and hands it to the external promotion controller.
- The external promotion gate remains: proxy gain at least 0.005, increased
  division true positives and Jaccard, adjusted-edge regression at most 0.001,
  no public-output hash match, and exact additive/no-reassignment invariants.
- Relevant graph/relational controller and candidate tests: 31 passed.

This model is an experiment, not a candidate. It has no submission authority
until it passes movie-disjoint selection, sealed audit, frozen development, and
the external candidate promotion gate.

## 2026-08-30 — topology-preserving temporal localization

The frozen four-movie control audit contains 2,357 annotated nodes and 2,284
matches (0.969028 recall). Of the 73 misses, 68 already have a prediction within
5–12 µm, two are one-to-one crowding conflicts, and only three lack a nearby
prediction. The median/p90 nearest-prediction distances among misses are
6.563/9.283 µm. Detection replacement and node addition therefore target the
minority failure mode; coordinate localization is the new primary branch.

- Model: project-authored temporal ConvNeXt3D plus axial morphology and 12
  frozen graph-motion features, 71,249,805 parameters per member.
- Mechanical scope: bounded coordinate offsets only. Node IDs, counts, times,
  and lineage edges are immutable.
- Data: the CC0 Synthetic16 subset, 27,696 labeled nodes in 16 six-frame
  sequences. Sequences 0–11 train, 12–13 select checkpoints, and 14–15 remain
  unopened until each checkpoint is serialized and hash-frozen.
- Schedule: four independent seeds on four AWS A10Gs, 20,000 steps per member,
  equal-weight ensemble only if every included member independently passes the
  selection and sealed-audit gates. Audit subset/weight searches are forbidden.
- AWS controller PID: 44716 (the initial 27152 controller was safely replaced
  before launch to add the frozen real-domain development gate). The validated
  launch target is one temporary
  `g5.12xlarge`; it harvests a SHA-256-verified result archive and auto-stops
  after harvest or after a bounded grace period. At launch time the CLI profile
  still returned `ExpiredToken`, so the controller is credential-event-driven.
- After every member's synthetic checkpoint is selected and sealed-audited, the
  precommitted consensus is evaluated on five cached center frames from the
  already-opened four-movie real probe. It requires an added match, lower mean
  matched distance, no per-movie match regression, and the global move-fraction
  gate. This is development evidence only and supplies no submission authority.
- This stage reads no competition images, labels, predictions, or leaderboard
  values and cannot create a submission.

### Temporal-localization candidate chain

- A harvested ensemble is packaged only if three or four independently accepted
  71,249,805-parameter members and the frozen real-development gate all pass.
  Archive extraction is path-safe and bound to the remote SHA-256 receipt.
- The private Kaggle notebook retains the attributed clean control and adds only
  bounded consensus coordinate corrections. It requires exactly two T4 GPUs;
  node identity, node count, times, and all lineage edges remain immutable.
- The Kaggle watchdog is capped at a 39,600-second declared budget with a
  38,400-second hard stop. With 19.14 GPU hours currently remaining, the launch
  gate reserves at least 8.0 hours even if the full declared budget is consumed.
- External promotion requires proxy gain at least 0.001, fewer missed ground-
  truth nodes, no per-movie miss regression, no spurious-node increase, division
  Jaccard non-regression, adjusted-edge regression no worse than 0.001, and a
  submission hash distinct from all audited public outputs.
- Only that external report authorizes a single rate-limit-checked submission.
  Kaggle leaderboard values are not read for selection or promotion.
