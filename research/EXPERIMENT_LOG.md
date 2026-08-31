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
- Data: a provenance-bound CC0 Synthetic256 subset with 480,273 labeled nodes
  in 256 six-frame sequences. Sequences 0–239 train, 240–247 select
  checkpoints, and 248–255 remain unopened until each checkpoint is serialized
  and hash-frozen.
- Schedule: four independent seeds on four AWS A10Gs, 20,000 steps per member,
  equal-weight ensemble only if every included member independently passes the
  selection and sealed-audit gates. Audit subset/weight searches are forbidden.
- AWS controller PID 44716 was safely stopped before any launch terminal or EC2
  allocation so the division-critical and exact-serialization gates could be
  added to the frozen contract (the initial 27152 controller had likewise been
  replaced before launch to add the real-domain development gate). The validated
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
- External promotion requires proxy gain at least 0.005, fewer missed ground-
  truth nodes, no per-movie miss regression, no spurious-node increase, division
  Jaccard non-regression, adjusted-edge regression no worse than 0.001, and a
  submission hash distinct from all audited public outputs.
- Only that external report authorizes a single rate-limit-checked submission.
  Kaggle leaderboard values are not read for selection or promotion.

### 2026-08-30 live research refresh and division-critical gate

- Live snapshot `e1fb78def1c9` retrieved all competition pages, recent topics,
  quota, and the policy-required 50 scored public notebook sources. Every rules, code-requirements,
  evaluation, data, description, prize, and timeline fingerprint is unchanged
  from the 2026-08-28 snapshot. Ten explicit metric-hack notebooks remain
  excluded.
- The newly rerun 0.931 notebook is 0.952 normalized code-line Jaccard with the
  pinned 0.940 EMA control. The new `edgebar040` notebook is 0.954 Jaccard with
  that 0.931 notebook. Both are public-lineage configuration variants, not new
  architectures and not candidate outputs.
- The independent Detector3D notebook contains a temporal 3D U-Net and four-view
  Y/X flip TTA. Its embedded source compiles and matches no exploit signature,
  but its private checkpoint is not reproducible. Detection replacement remains
  lower priority because the frozen residual audit found only three misses with
  no nearby prediction.
- Discussion topics 737543, 737101, and 734604 converge on complete-movie
  validation and bottleneck decomposition; they specifically report that high
  global node recall does not guarantee division quality and that parent/daughter
  localization near 3 um matters. This supports a model gate, not copying a
  public threshold or prediction.
- Synthetic256 contains 45,211 division-critical interior nodes among 319,623
  eligible nodes; every sequence contributes at least 76 division-critical
  rows. The heavy trainer
  now forces four of each 16 batch slots from parent/daughter rows and requires
  both global and frozen division-critical selection/audit gains for every
  deployable member.
- The exact FP16 checkpoint is hash-frozen and must re-pass both selection strata
  before the sealed audit files are opened. This closes pre-/post-serialization
  evidence drift while preserving the four-seed equal-weight policy.
- Final pre-launch regression: 31 temporal-localization tests passed. The
  controller validation binds runner SHA-256 `02b98e20e0e23b1fd87f4c06a1b6769184d825e0d468b2c4aed32878f23e4ae8`,
  trainer SHA-256 `41d8ca55448aa2b6387ddf41693f722ac68d84b458f39fe60c62b952bca16be7`,
  Synthetic256 archive SHA-256 `d8eca77fcaabaad185bfeaa20351a5afec1f2597106822525ade35b40d43a095`,
  and real-development archive SHA-256 `863d3edcce206266bfb6ad4d78afd033b662d1c57e3b0c11a81bb4bfc84a8b26`.
- Controller PID 46440 was stopped before launch while both terminal records
  were absent so the CUDA-environment bootstrap could be hardened. Dependency
  installation now detects whether user-site packages are visible instead of
  unconditionally using `pip --user`, verifies imports after installation, and
  records Python, package, CUDA, cuDNN, and all four GPU identities in the
  harvested runtime environment manifest. Controller PID 40268 and downstream
  PID 2852 were subsequently stopped before either terminal existed so the
  evidence contract could expand from 16 to 256 source sequences.
- Executable CPU smoke evidence used a real division-critical row from sealed-
  selection sequence 240: physical patch shape 1x3x17x17x17, graph shape 1x12,
  71,249,805 parameters, finite outputs, and two independent reloads of the
  142,589,136-byte FP16 checkpoint produced bit-exact outputs.
- The enlarged shard was downloaded file-by-file from the audited CC0 kernel.
  Current full-manifest SHA-256 `e8b5376b2ac6fdd55bd6e45d1b07b401339d375b211b6f67be93fb0de4d8ce14`,
  metadata SHA-256 `328b9bb2545309e545cf68663ef985034ec68c36c0b709408a0f91801fedf89e`,
  and overlapping sequence 0000 all exactly match the prior source audit. The
  launcher additionally checks every selected file against the audited byte
  inventory before creating the 753,671,907-byte deployment archive.
- Synthetic256 AWS launch/harvest controller PID 7612 and downstream candidate
  controller PID 48392 were armed after commit `f392008`. Both started with no
  launch, harvest, or candidate terminal present. The AWS controller waits for
  the credential event, allocates at most one four-A10G instance, and the
  downstream controller remains unable to launch Kaggle or submit unless the
  complete synthetic, serialized-checkpoint, real-development, and external
  promotion gates all pass.

## 2026-08-30 - Synthetic256 plus train-only real localization replay v2

The synthetic-only localizer was not launched. Before allocating AWS GPUs, the
training contract was expanded with fixed, competition-train-only image replay
so every large member must learn and improve on the real microscopy domain as
well as the synthetic generator.

- The source inventory is SHA-256
  `55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1`:
  115 movies, 177 center-frame triplets, 525 required image frames, and 1,622
  annotated center nodes. It contains 146 optimization, 17 selection, and 14
  sealed-audit triplets.
- Every source path is under competition `train/`. The four previously opened
  development probes (`44b6_12dfb391`, `44b6_267148e4`, `6bba_062c8d37`, and
  `6bba_07e24132`) are excluded. Competition test data and leaderboard values
  are not read.
- Geometry was checked before scale-up on 46 annotated nodes: all labels were
  in image bounds, mean truth-voxel frame percentile was 0.984356, and the
  minimum was 0.975874. An executable native-to-pooled triplet passed through
  `corrected_sequence_sample` and `build_sequence_state` with shape
  `3 x 64 x 64 x 64`, eight graph nodes, two center nodes, and one
  division-critical center node.
- The resumable frame cache reached 495 of 755 required metadata/chunk objects
  (1,513,104,515 bytes) before Kaggle returned a persistent HTTP 429. Fetching
  is now gate-first after cooldown, paced at five seconds per new request, and
  uses a bounded 120--300 second rate-limit backoff. Existing hashes are reused.
- Training run ID is
  `synthetic256-real-replay-temporal-node-localizer-v2`. Each step draws from
  real triplets with fixed probability 0.25 and otherwise from Synthetic256;
  global and forced division-critical examples are sampled within the selected
  domain. Non-division real movies contribute to the global pool while the
  division pool remains strictly event-positive.
- Checkpoint promotion requires improvements in four strata: synthetic global,
  synthetic division-critical, real global, and real division-critical. The
  exact serialized FP16 checkpoint must re-pass all four before either sealed
  synthetic or sealed real audit features are loaded. Candidate packaging,
  runtime verification, and notebook startup independently recheck these gates.
- The scheduled AWS job is four independent 71,249,805-parameter
  ConvNeXt3D/axial members, one per A10G on a single `g5.12xlarge`, 40,000 steps
  per member (160,000 aggregate model-steps), fixed equal-mean ensemble, and no
  member-subset or weight search. Validation runs every 2,000 steps; the
  per-member wall cap remains 10.5 hours so outputs are finalized safely.
- A bounded overnight orchestrator waits out the Kaggle throttle, completes and
  hashes the real replay shards, then hands off to the AWS credential-event
  controller. The AWS profile still returned `ExpiredToken` at 2026-08-30
  06:15 CDT, so no EC2 instance had been allocated at that point. Kaggle GPU
  quota remains untouched.
- Focused regression evidence before arming: 16 trainer/replay tests, 20
  candidate/runtime tests, and 16 AWS/controller tests passed in their
  respective suites. The experiment remains unauthorized for submission until
  every scientific and external candidate gate passes.
- After commit `b35d30d`, overnight replay/build/AWS controller PID 20816 and
  downstream scientific-promotion controller PID 47640 were armed. Both were
  alive with empty error logs at startup; the first observes a 900-second
  Kaggle cooldown before resuming, and the second can only advance after a
  hash-verified AWS harvest terminal exists.
- After the cache audit proved 495/755 objects intact and the prior HTTP 429
  window had aged by more than 20 minutes, sleeping PID 20816 was safely
  replaced before any network or AWS child existed by PID 48740 with 120
  seconds of remaining cooldown. The five-second request pace and bounded
  retry policy are unchanged.
- A pre-launch feature audit found and removed a synthetic shortcut: target
  proposals were jittered while truth parent/child coordinates stayed fixed,
  allowing relative graph vectors to encode the correction directly. Training
  now co-translates local lineage and same-frame context with the proposal, so
  motion/density features are invariant to injected target jitter and image
  appearance must explain the offset. A deterministic invariance regression
  covers all graph features except the intentionally position-dependent image
  boundary feature.
- Training now also applies independent 50% reflections on each physical axis,
  with the image crop, offset target, parent vector, and child vector reflected
  together. This expands orientation coverage without changing temporal order,
  physical scale, lineage flags, density, or boundary distance; deterministic
  tests verify both reproducibility and vector-sign consistency.
- Real division event shards contain the dividing parent at their center and
  both daughters in the final local frame. The real rare-event sampler now
  retains all three rows and boundary-clamps only the daughter's unavailable
  future image channel; ordinary real sampling remains interior-only. Thus the
  real selection and sealed-audit division gates cover both parent and daughter
  localization rather than silently testing parents alone.
- The 40,000-step scratch schedule now warms linearly from `2e-6` to `2e-4`
  over 1,000 steps before cosine decay back to `2e-6`. Learning rate is applied
  before each optimizer update, avoiding the prior full-rate first step on an
  untrained 71.25M-parameter network.
- Because hundreds of competition-file API requests remained rate limited, a
  second, CPU-only materialization path was added without changing the replay
  inventory or scientific split. A private 866,075-byte train-label bundle
  (`biohub-real-localization-labels-v1`, version 1) is bound to archive SHA-256
  `94bee4145a11d4da8bc958833b3afbf2f3986464428f201de640b62c477d8353`
  and label-manifest SHA-256
  `8384d402874f47bfc4288f8df028310e0cf6dfd30958999679d369770b68c872`.
  It contains only graph labels for the same 115 train movies; the frozen four
  development probes remain excluded.
- Private Kaggle kernel `biohub-real-localization-replay-cache-v1`, version 1,
  was launched with CPU only, GPU/TPU/internet disabled, and only the label
  dataset plus official competition input attached. It resolves exactly one
  `train/` root, reads the same 525 frames, emits the same 177 pooled replay
  shards, records per-frame content digests, and cannot submit. A background
  harvest controller will validate every output hash and atomically prefer the
  first complete replay path before handing off to the existing AWS launcher.
- The Kaggle label dataset was ready before the kernel launch. A fresh AWS STS
  check at the same handoff still returned `ExpiredToken`; therefore no AWS
  instance or GPU had been allocated, and the existing credential-event wait
  remains the launch boundary.
- CPU cache kernel version 1 failed closed after 9 seconds because the base
  image did not contain `zarr`; it read no competition frame and consumed no
  GPU. Version 2 removes that undeclared dependency. The private label dataset
  now also carries the exact 8,846,228-byte CPython-3.12 manylinux `numcodecs`
  wheel (SHA-256
  `44869ef564a50aa545215c6a0d42ba5bbc34e9715523fb2336ada3d1fb2b331d`).
  The notebook installs it offline with `--no-index --no-deps`, validates the
  exact Zarr-v3 bytes+Blosc-Zstd/bitshuffle metadata contract, and decodes each
  chunk directly. The decoder was checked byte-for-byte against local Zarr on
  a real cached competition frame before the version-2 launch.
- Kernel version 2 also failed closed in 10 seconds before reading a frame:
  Kaggle retained the original version of the already-attached v1 label
  dataset, so the new support wheel was not present in that run. Kaggle also
  expands uploaded tar archives under their archive-stem directory. To remove
  both ambiguities, version 3 attaches a newly created private dataset slug,
  `biohub-real-localization-labels-v2` version 1, and uses the exact mounted
  dataset and archive-stem paths verified from its 119-file remote inventory.
  Dataset and kernel remain CPU-only, offline, train-only, and unable to
  submit; prior failed versions consumed neither Kaggle nor AWS GPU time.
- The downstream candidate handoff was hardened before any AWS harvest. The
  obsolete eight-hour Kaggle reserve was changed to zero to reflect the later
  authorization to use all available quota; the 39,600-second dual-T4 runtime
  requirement itself remains enforced. Kaggle's private-kernel API returns 403
  for a not-yet-created slug, so authenticated owner-scoped exact listing now
  proves absence without guessing a version, while post-push state verification
  uses bounded retries and still fails closed for a present-but-unreadable
  kernel.
- Before the harvest terminal existed, waiting downstream controller PID 47640
  was replaced by PID 6652 with the updated code and an explicit zero-hour
  reserve. It remains event-driven and cannot allocate Kaggle GPU, build a
  candidate, or submit until the AWS scientific gates and external frozen-probe
  promotion gate pass.
- After refreshed credentials exposed the already-running Antelume
  `g5.xlarge`, a read-only recovery audit found that the failed four-GPU
  temporal-localizer launcher had never allocated an instance and therefore
  produced no resumable localization checkpoint. The Antelume volume does
  retain eight completed 48M relational models, but their frozen deployment
  policy failed audit, so they remain research-only. A later graph-context
  sweep retained five completed 299,026,046-byte models and stopped during the
  sixth member at step 18,200 without a sixth checkpoint or optimizer state;
  the five complete models are preserved for a future precommitted diversity
  evaluation and are not promoted by inspection.
- The running instance has one idle 23,028 MiB NVIDIA A10G, not four GPUs. The
  unchanged four-seed, 40,000-step, 71,249,805-parameter localizer schedule is
  therefore deployed sequentially, with a 16,500-second member training cap
  and best-checkpoint retention. Development inference now deterministically
  maps accepted members round-robin over any positive visible CUDA count, so
  all four checkpoints can be scored on the one A10G without changing the
  equal-member consensus or any scientific gate. The instance is explicitly
  preserved after completion, and a hash-bound background harvest hands the
  result to the existing candidate gate and submission controller.
- The first one-A10G launch failed closed before step 1 for all four seeds:
  PyTorch 2.5 rejects probability-form `binary_cross_entropy` inside CUDA
  autocast. The generated result archive was only 40,026 bytes and contained
  no model; the downstream controller was stopped before consuming it. The
  calibration term now exits the surrounding AMP region and computes BCE in
  FP32 while retaining the probability-valued inference interface. A CPU
  autocast regression asserts both properties before the clean relaunch.
- A separately launched RSNA knee-model process was found sharing the A10G
  during initial throughput calibration; it consumed about 5 GiB and compute.
  The user had explicitly authorized clearing RSNA work, so that process alone
  was stopped without deleting its files. On the uncontended A10G, a measured
  100-step interval took 80 seconds (about 4,500 steps/hour). The final bounded
  schedule therefore retains all four seeds and raises the per-member cap to
  16,500 seconds with a 900-second finalization reserve: about 18,000--19,000
  realized steps per member and roughly 19 aggregate GPU-hours by morning.
- The RSNA automation then relaunched as transient user services
  `plw7.service` and `bigmembers.service`, consuming 5 and 14.5 GiB of GPU
  memory respectively. Both service names were stopped and runtime-masked for
  the Biohub run under the user's explicit RSNA cleanup authorization. This is
  reversible (`systemctl --user unmask`) and deleted no RSNA files. Final
  Biohub runner PID 11391 passed its first optimizer step with the A10G as its
  sole workload; hash-bound harvest PID 49508 and gated candidate/submission
  PID 41512 remain event-driven rather than polling the GPU run externally.
  Detached lease-restorer PID 11654 removes both runtime masks after
  `run.complete`, or after a bounded 21-hour lease, without starting either
  RSNA service.
- The 2026-08-31 authenticated public-source refresh classified C33, C34, and
  C35 as comparators rather than independent model candidates. Their normalized
  code-line overlap with the pinned public 0.940 EMA notebook is
  `0.935484`, `0.928000`, and `0.926716`; C34 and C35 overlap each other at
  `0.995869` and differ only in a small set of labels, scalar settings, and
  guards. None contributes a new detector or linker architecture, so none is
  copied into the active candidate.
- The same refresh retained only non-exploit geometry findings from the public
  metric-analysis notebook: sparse labels, costly duplicates, micron-scale
  localization sensitivity, and embryo-disjoint evaluation. Metric edge cases,
  count bonuses, fabricated nodes, and fake division constructions remain
  excluded from training, selection, and submission. Recent discussions
  reinforce component-wise learned improvement—detection, linking, then
  division—and report that crowded-frame divisions are not solved reliably by
  standalone hand rules. Full evidence is recorded in
  `research/PUBLIC_NOTEBOOK_AUDIT_2026-08-31.md`.
- The interrupted graph-context sweep contained five complete, hash-bound
  74,732,308-parameter checkpoints, not merely logs. All five passed the
  movie-disjoint selection gate: pooled AP ranged from `0.870487` to
  `0.928614`, and each recovered 9--11 true positives before the first false
  positive. Their worker terminals prove that sealed audit, final probes,
  competition test data, and submission were never opened. The sixth worker
  stopped at step 18,200 with no checkpoint or optimizer state and is not
  treated as resumable evidence.
- Commit `3e0004b` adds a fail-closed recovery mode. It admits only complete
  workers whose checkpoint, initialization, architecture, selection history,
  terminal, and pre-audit flags all revalidate. The five source directories are
  hard-linked into an isolated result tree while the interrupted source tree is
  preserved byte-for-byte. The sixth and final two members train from scratch;
  only after all eight terminals exist may the original selection policy freeze
  an individual or equal-rank ensemble and open sealed audit once. Recovery
  provenance is required by the streamed archive verifier.
- Antelume recovery runner PID 14149 is queued server-side behind the active
  temporal localizer and an idle-A10G check. Local hash-bound harvest,
  development, and candidate/promotion controllers are PIDs 37672, 46568, and
  45636. None can submit without the original sealed-audit, exact development,
  and external candidate gates.
- A system-level transient `rsna-plw.service` relaunched the unrelated RSNA
  process despite the two user-service masks. It was stopped and runtime-masked
  without deleting files. Lease-restorer PID 14997 now holds all three reversible
  masks through both Biohub jobs and un-masks them afterward; it never starts an
  RSNA service.
- The first full four-stratum localizer validation completed at step 2,000. It
  passed synthetic global (`+33.29%` mean-residual gain), synthetic
  division-critical (`+27.16%`), and real global (`+21.73%`) gates. Real
  division-critical localization improved mean residual by `15.69%`, p90 by
  `0.407156` microns, and within-5-micron recall by `0.125`, with every axis MAE
  non-regressive; only its precommitted `20%` mean-improvement condition remains
  unmet. Its mean residual is `3.455568` microns versus a `3.278823` gate, a
  further `0.176745` microns (`5.11%` of the current residual). This is promising
  early evidence but not an accepted checkpoint, so training continues and no
  audit or candidate has opened.
- A second system transient, `rsna-dump.service`, later started an unrelated
  RSNA prediction-dump process on the leased A10G. It was stopped and
  runtime-masked without deleting files. Commit `0d28a33` replaces the static
  lease restorer with a bounded dynamic guard: while either Biohub job remains
  alive it detects running system services whose exact names match
  `rsna-*.service`, stops and runtime-masks only those units, records every
  affected name, and releases all recorded masks after both jobs. The deployed
  script hash is
  `cedabdf329f9ce7fa368b5051950e25371f7c903c5300e8653b78aba52ded66d`;
  guard PID 16882 is live, the localizer is again the sole GPU process, and the
  graph-context recovery remains queued behind it.
- The user then clarified that RSNA is intentionally sharing the Antelume
  instance and must not be disrupted. That instruction supersedes the earlier
  cleanup authorization. Guard PID 16882 was terminated, all two user-level and
  two system-level runtime masks were removed, and the RSNA prediction process
  resumed alongside Biohub. Commit `f352962` converts the deployed helper into
  a passive observer with no `systemctl`, stop, mask, unmask, start, or delete
  capability. Its deployed SHA-256 is
  `b88745324dec860cc22c40ba361f4241cffe24466acf810b67280ce13323186d`.
  From this point Biohub may share available capacity or wait for idle capacity,
  but it must never manage an RSNA process or service.
- Commits `9483673` and `45ba339` precommit a conditional composition of the
  two independent candidate families. It applies graph-context division first
  so that its frozen geometry policy sees the original EMA coordinates, then
  applies temporal localization without permitting further topology changes.
  The composition cannot launch until the graph and localization candidates
  each have their own hash-bound `eligible_for_submission` report. It cannot be
  promoted unless it retains the graph candidate's division quality, is
  per-movie non-regressive to the localization candidate on missed and spurious
  nodes, passes the clean public-control gate, and improves exact proxy score by
  at least `0.001` over the stronger standalone component. No component subset,
  threshold, blend, or leaderboard result selects the composition. The private
  two-T4 notebook contains no submission command; only the external verifier
  can authorize the one-shot submitter. Event-driven controller PID 23688 is
  waiting for both standalone promotions and has no AWS, service, or RSNA
  control surface.
