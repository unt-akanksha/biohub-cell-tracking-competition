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
- Composition verification was strengthened in commits `0a6ed58` and
  `099d255`. A complete synthetic runtime fixture now exercises the actual
  attributed-notebook transformation and proves that the two distinct manifest
  hashes are embedded with graph-before-localization ordering and no submission
  command. A full external-verifier fixture then proves positive promotion only
  when both standalone reports are present and the composed exact metric beats
  both; the same fixture rejects a composition that merely ties the stronger
  graph component. The broader graph, localization, and composition regression
  selection passed 41 tests before these two end-to-end additions.
- At step 4,000, temporal-localizer seed 41021 remained just below the frozen
  four-stratum selection gate but improved monotonically on the only failing
  stratum. Real division-critical mean-residual gain rose from `15.6876%` at
  step 2,000 to `17.2199%`; mean residual fell from `3.455568` to `3.392767`
  microns. The unchanged 20% boundary is approximately `3.278823` microns, so
  the remaining gap narrowed from `0.176745` to `0.113944` microns. Synthetic
  global (`39.09%`), synthetic division-critical (`32.43%`), and real global
  (`24.82%`) all passed, every real-division axis remained non-regressive, p90
  improved by `0.407516` microns, and within-5-micron recall improved by
  `0.117188`. No checkpoint or audit is admitted yet because the aggregate gate
  is still false; the precommitted schedule continues unchanged.
- Commit `d56822f` assigns the conditional two-component notebook the full
  12-hour Kaggle runtime envelope (`43,200` seconds) with a 20-minute
  (`1,200`-second) finalization margin and watchdog trigger at `42,000`
  seconds. This changes no model, input, threshold, validation criterion, or
  submission authorization; it only avoids inheriting the shorter standalone
  localizer envelope for a notebook that must run both independently promoted
  stages. All 13 composition builder, verifier, controller, and submitter tests
  passed after the change.
- An audit of live autonomous processes found one legacy
  `wait-build-verify-submit-learned-division-candidate.ps1` controller still
  polling after roughly 33 hours. Its required
  `/home/ubuntu/biohub-results/division-policy-v1/policy.json` is absent from
  the active Antelume volume, its embedded remote address is obsolete, and it
  had produced no policy, runtime, candidate, promotion, or submission receipt.
  PID 16300 was therefore retired to prevent an unverifiable older lane from
  later consuming a Kaggle run or submission slot. No remote process, GPU job,
  AWS service, file, or RSNA workload was changed.
- The shared Antelume boundary was strengthened with a Biohub-only yield guard.
  It classifies GPU clients by exact process ID, working directory, and one of
  the two approved Biohub trainer entry points. If any other GPU client appears,
  it sends `SIGSTOP` only to those verified Biohub trainer PIDs, records the
  exact PIDs it paused, and sends `SIGCONT` only after the unrelated client has
  released the GPU and each recorded PID has been reverified as Biohub. It has
  no service-management commands and never signals, restarts, masks, or deletes
  the unrelated workload; external process details are not persisted. The two
  safety tests
  plus the existing passive-observer test pass. The deployed script SHA-256 is
  `5d5f00dfc8c5db385bcd65737500ed6ee2a9438b1624a55d821e1b076a280097`;
  guard PID `23076` is live. Its initial state is `biohub-allowed` because the
  read-only GPU inventory contained only temporal-localizer PID `11505` using
  3,234 MiB and no unrelated GPU client at the observation time.
- A second authenticated 2026-08-31 top-50 source audit found no new strong,
  independent candidate. The late Evgen `0.933` source is `0.940597` code-line
  Jaccard with the attributed public control. Flexon and Rishabh are each about
  `0.917` versus that control and `0.998238` with each other. They are excluded
  as public-lineage replicas/configuration variants. Yusuke's ranker/look-ahead
  source is more modified (`0.474286`) but retains public detector/support
  assets; the owned graph-context lane already targets the same ambiguity with
  broader learned context. Xiaolei's three-branch four-flip 3D U-Net is
  independent (`0.003255`) but reports only `0.8623` on its own 24-movie
  validation and is not individually strong enough for the intended ensemble.
  Static scanning found no known exploit signature, but no displayed score was
  used as selection or clean-score evidence.
- The Antelume runner's post-training development-probe status initialized to
  `5`, which would incorrectly record a failure when the probe command returned
  zero. The local future-run template now initializes it to `0` and preserves
  the actual nonzero code through the existing guarded assignment. The active
  remote run was deliberately not modified, so its model recipe and provenance
  remain byte-stable; candidate admission continues to use the hash-bound JSON
  evidence rather than this diagnostic exit-code file.
- On 2026-09-07 the refreshed Antelume volume supplied definitive temporal
  localizer evidence. Seed `41021` completed 5,376 steps but was
  `rejected_at_selection`: the step-4,000 checkpoint passed synthetic global,
  synthetic division-critical, and real global gates, improving real global
  mean residual by `24.8201%`, but real division-critical mean gain was only
  `17.2199%` and its gate remained false. No checkpoint was frozen, no sealed
  audit was opened, and no submission was created. The partial seed `41029`
  reached only step 500 and is not evidence. The temporal-localization lane is
  therefore rejected rather than rescued or weakened after seeing results.
- Five graph-context checkpoints survived the Antelume restart and match the
  SHA-256 values in their selection terminals. All five independently passed
  the precommitted selection gate at exactly 20,000 steps. Their selection APs
  are `0.928614`, `0.876737`, `0.919652`, `0.870487`, and `0.888160`; their
  true positives before the first false positive are 10, 10, 10, 11, and 9.
  The recovery runner hardlinks those immutable members, validates their hashes,
  and trains only the three missing seed/initialization combinations. It was
  launched on the idle Antelume A10G as PID `3039` under Biohub-only yield guard
  PID `2811`. The guard's deployed SHA-256 is
  `2a1f2a1199e5da082e7ceb7b2934e7197f7567816a269bd71834d58726bbdd6d`;
  it reads a command line only after a GPU PID's working directory is proven to
  be one of the owned Biohub roots and never signals an unrelated process.
- The repaired graph pipeline is autonomous and event-driven. The stale August
  31 SSH-failure terminals were preserved with `.failed-20260831` suffixes, and
  new harvester, development-evaluator, and candidate-launch controllers were
  started with PIDs `48460`, `15000`, and `37860`. They may package and launch a
  private two-T4 Kaggle candidate only after hash verification and positive
  complete-movie development evidence; submission still requires the separate
  promotion verifier.
- A 2026-09-07 authenticated source refresh found that the new advertised
  `0.942`–`0.948` notebooks remain one public family. Pairwise normalized-code
  overlap is `0.91`–`0.99` for the central cluster, including `0.969447`
  between the `0.948` TTA and `0.942` one-knob notebooks and `0.988364` between
  Rishabh's detector-fusion variants. They are excluded as replicas or
  configuration variants. Muhan's author-finetuned edge checkpoint plus D4
  edge TTA is more modified but still uses the public inference architecture;
  Hengck's three-level 3D U-Net point detector is independent but only a
  one-movie demonstration. Static scanning found no registered exploit
  signature, which is not a reproduced score claim. No public score, output, or
  displayed prediction is used for selection.
- Recovered Kaggle execution logs close the earlier detector branches without
  relying on notebook titles or displayed scores. Spotiflow PU reached only
  `0.779239` pooled annotated recall and `0.486438` worst-movie recall.
  SpatialDINO selective distillation reached `0.885906` and `0.602170`; LSM-FM
  PU reached `0.884206` and `0.622061`; the image-text LSM-FM variant reached
  `0.889128` and `0.641953`. All failed their frozen worst-movie selection gate
  and never opened acceptance. Learned center offsets, feature-36 coordinate
  refinement, and probability ensembles also failed per-movie non-regression.
  A predeclared LSM-FM refinement of public coordinates raised four-movie
  pooled recall from `0.969028` to `0.970725`, but regressed one movie by
  `0.002786`, so it was correctly not promoted.
- The next independent detection experiment is
  `synthetic256-real-positive-temporal-peak-rank-v1`. It is not a port of a
  public notebook: a 38,381,478-parameter temporal 3D ConvNeXt/U-Net consumes
  previous/current/next frames plus explicit differences, predicts dense peaks
  and subvoxel offsets, and uses a bounded hard-neighbor ranking loss that does
  not label the unannotated real volume as background. Synthetic256 indices
  0--239 optimize, 240--247 select, and 248--255 stay sealed; 146/17/14
  stem-disjoint real positive-only shards have the same roles. The sealed audit
  is opened only after synthetic AP/recall and real positive-peak localization
  gates all pass. A fixed 12,000-step A10G run is queued behind graph recovery;
  the Biohub-only yield guard will pause only this exact trainer if any
  unrelated GPU workload appears.
- The peak-ranking lane now has a fail-closed autonomous handoff. Harvester PID
  `44584` waits for the immutable AWS archive and verifies every archived file,
  the frozen 38,381,478-parameter/12,000-step contract, the checkpoint hash,
  and sealed-audit state. Validation controller PID `21972` does nothing after
  a training rejection; after a positive audit only, it packages a private
  runtime and runs an offline two-T4 Kaggle screen. Eight selection movies are
  evaluated first across isolated GPU workers; the four acceptance movies stay
  closed unless pooled and worst-movie selection gates pass. The validation
  kernel contains no submission command or candidate materialization path.
- A late September 6 Kaggle refresh audited five additional hot notebooks from
  Aman Atar, Analytica Obscura, Anhad Mahajan, Kunal Desale, and Nusrati. Their
  closest normalized-code overlap with the already-audited public family is
  `0.921549`--`0.990681`; all retain the same U-Net/Trackastra/ILP inference
  structure rather than supplying an independently validated model. Static
  scanning found only explicit `metric_hack_used = False` text, not a known
  exploit signature, but that is not proof of clean evaluation. They remain
  excluded from candidate construction and no displayed score or output is
  treated as experiment evidence.
- The peak-ranking detector now has a tested association bridge ready for a
  positive validation result. It performs complete-movie temporal inference,
  fixes one density threshold from organizer-provided node-count metadata,
  substitutes only the official linker's detector callback, and restores the
  detector's sub-voxel coordinates afterward. Thus an accepted checkpoint can
  become a full tracking candidate without copying a public prediction or
  retraining/tuning the association model on leaderboard feedback. Fifteen
  focused bridge, inference, and identity-preservation tests pass.
- The recovered eight-member graph-context sweep completed on 2026-09-07 and
  was rejected at its sealed audit. Its equal-rank selection ensemble reached
  AP `0.951062` with 11 true positives before the first false positive, but the
  precommitted strongest-selection deployment member (`seed-813121-init-2`)
  failed audit. Two different members passed audit, but selecting either after
  observing audit labels would violate the frozen policy; consequently
  `ensemble_eligible=false`, no final development probe was opened, and no
  submission was authorized. The queued peak-ranking trainer then started on
  the Antelume A10G and held about 6.9 GB at its initial data-loading phase;
  the Biohub-only yield guard remained in `biohub-allowed` mode.
- A label-free production worker is now staged for any promoted peak-ranking
  checkpoint. Two isolated Kaggle processes partition the complete test-movie
  inventory without overlap, each exposes exactly one GPU, generates node
  locations with the independent detector, and routes those nodes through the
  audited official association model and ILP. Worker manifests bind the model
  hash and explicitly record that no train labels, test labels, public
  predictions, leaderboard selection, or submission creation occurred. Twelve
  focused production, bridge, packaging, validation, and controller tests pass.
- The peak-ranking lane now has an end-to-end autonomous candidate path using
  the audited 0.948 public code only as an attributed association/ILP/finishing
  backbone; its advertised score is explicitly not treated as evidence. Both
  test and held-out inference are replaced by the same independent detector.
  A promoted runtime is versioned only after clean two-GPU validation, the full
  candidate again requires two GPUs, and the external gate was frozen before
  results at proxy gain `>=0.003`, aggregate adjusted-edge delta `>=-0.001`,
  and worst-movie proxy delta `>=-0.005`. An exact public-output hash is
  rejected. Only the separate one-shot submitter may cross the competition
  boundary after all gates pass.
- Before the first peak-ranking selection evaluation, a live throughput audit
  showed that the original 12,000-step declaration required roughly 20+ hours
  while its hard wall guard was 7 hours, making a valid terminal impossible.
  With no selection metric yet observed, the infeasible run was terminated at
  step 250 and preserved as a non-evidentiary archive. The contract was changed
  for resource feasibility—not performance—to 3,000 cosine-scheduled steps,
  with the same 38,381,478-parameter model, data partitions, seed, objective,
  and frozen evaluations at steps 1,000, 2,000, and 3,000. At the observed
  throughput this fits the guard while using the A10G at 100% compute.
- A September 7 incremental public refresh found one new frontier source,
  `redoctopusk/biohub-948tta2`. Its normalized-code overlap with the pinned
  `biohub-948tta` source is `0.994411`, so it is not an independent model and
  neither its prediction nor advertised score is evidence. Its sole material
  change averages secondary-detector association features across D4 views by
  reusing forward passes already performed for detector TTA. Before any
  peak-ranker selection result was observed, the full candidate builder was
  hash-pinned to this attributed, compute-neutral linker update. The independent
  detector and all frozen clean-validation/promotion gates remain unchanged.
- Before v1 produced its first selection result, a complementary detector run
  was frozen and queued unconditionally behind it. The v2 member retains the
  38,381,478-parameter temporal ConvNeXt/U-Net and complete-label synthetic
  objective, but sparse competition crops contribute only positive-logit and
  subvoxel-offset loss; they contribute no negative rank term. It also applies
  random axial intensity attenuation down to `0.25`, matching published 3D
  microscopy augmentation for depth-dependent fluorescence decay. The run uses
  seed `1407733`, the same disjoint splits and gates, 3,000 steps, and the same
  7-hour wall guard. It starts only after the v1 archive is hash-verified and
  the shared A10G is idle; the Biohub yield guard remains authoritative if an
  unrelated workload later appears.
- Before v1 reached its first step-1,000 selection checkpoint, the downstream
  inference policy was made runtime-aware without changing training. Two
  held-out calibration movies compare fixed `none`, `rot4`, and `d4` detector
  inference (1, 4, and 8 views). The fewest-view mode is selected only when its
  pooled recall is within `0.003` and its worst-movie recall within `0.01` of
  the best observed mode while also clearing the original absolute selection
  floors. The selected mode alone scores the other six selection movies and,
  only after a pass, the four sealed acceptance movies. This reuses calibration
  rows rather than rerunning them and records frame-normalized throughput.
- Production now hash-binds that clean-selected TTA mode into the promoted
  runtime and both test and validator worker commands. Each two-GPU worker
  records per-movie frames and elapsed time, projects its full shard runtime
  with a frozen `1.15` safety factor, and fails early above `31,500` seconds.
  This reserves 9,900 seconds inside the notebook's 41,400-second hard stop for
  graph finishing, validation, CSV creation, and evidence serialization. A
  candidate cannot be externally promoted unless the recorded worker runtime,
  projected runtime, selected mode, checkpoint, and clean-validation hashes all
  agree. Twenty-nine focused peak-rank tests pass after this change.
- At the same pre-selection checkpoint, the active Antelume v1 run remained
  healthy at step 200. Training loss moved from `4.120774` at step 50 to
  `2.496155` at step 200; the A10G process used about 6.9 GB and the v2 queue
  remained idle after verifying every required source hash. This is training
  telemetry only, not promotion evidence.
- A pre-run integration audit caught that the external-node worker loaded only
  the public primary association model even though the candidate contract
  attributed the two-model `948tta2` linker. The worker now independently
  loads both hash-recorded association checkpoints and fails closed unless the
  frozen public configuration is exactly present: secondary edge weight
  `0.20`, low-margin consensus, bidirectional weight `0.15`, edge threshold
  `0.48`, and primary plus secondary edge-feature TTA. These models generate
  edges only; the project detector still owns every node and subvoxel
  coordinate. Worker manifests record both association hashes and settings,
  and promotion requires the same evidence. Sixty-two focused peak-rank tests
  pass after this correction; no candidate had been launched under the faulty
  integration.
- The queued depth-robust PU v2 lane now has a distinct autonomous clean
  validation handoff. After its existing AWS harvester verifies the frozen
  archive, a separate private runtime dataset and separate offline dual-T4
  kernel run the identical 1/4/8-view policy, complete-movie selection, and
  sealed acceptance gates. The v1 runtime, kernel, and controller identities
  are unchanged, so v2 cannot overwrite or contaminate the earlier candidate.
  This lane still creates no submission; a v2 checkpoint must first earn clean
  validation evidence before any ensemble or candidate integration.
- A second fail-closed candidate controller is also prepared for v2. It waits
  for that distinct clean-validation terminal, binds the selected detector TTA
  and checkpoint into a separately named runtime and Kaggle kernel, runs the
  same complete-movie tracking promotion gate, and submits only when the
  non-replica/proxy/edge/worst-movie/runtime checks all pass. Builder, verifier,
  submitter, output cache, promotion receipt, and run IDs are variant-aware;
  the existing v1 process remains isolated. Sixty-five focused peak-rank tests
  pass across both autonomous paths.
- A September 7 source-level refresh audited the newly published
  `biohub-detfusion-sdw30-exact0943`, `biohub-detfusion-sdw80-exact0943`,
  `biohub-0-942-lb-one-knob-past-the-public-line`, and `biohub-run77`
  notebooks. The two detector-fusion notebooks have `0.999037` normalized
  line overlap and differ only in a secondary detector weight (`0.30` versus
  `0.80`). The advertised 0.942 notebook is `0.985431` line-identical to the
  already audited `948tta2` source and changes the detector threshold from
  `0.965` to `0.96` after explicit public-leaderboard probes. Those weights,
  thresholds, displayed scores, and predictions are excluded from selection.
  The general idea of detector fusion remains a hypothesis for independently
  trained members only, subject to the existing clean movie-held-out gates.
- Before v1 produced its first step-1,000 selection result, a third detector
  member was frozen for the sequential A10G queue. It widens the same
  independently authored temporal ConvNeXt/U-Net to
  `(128,256,512,1024)`, yielding exactly `66,977,670` parameters, while using
  the conservative positive-unlabeled real loss and depth attenuation from
  v2. Seed `2607157`, 2,000 steps, checkpoints at 1,000 and 2,000, and a
  25,200-second wall guard were fixed from the measured v1 throughput rather
  than any validation result. It is 74.5% larger than v1/v2 and exists to add
  capacity and ensemble diversity, not to reproduce a public notebook.
- The capacity member cannot start until v2 completes, both predecessor
  archives pass local hash verification, and the A10G is idle. Verified local
  harvests write remote acknowledgements; only then may the runner delete the
  exact redundant Biohub v1/v2 result directories and archives to recover disk
  space. Resolved paths are allow-listed, a 1.3 GB free-space floor is checked,
  and no unrelated project path or process is inspected or changed. The v3
  archive, private two-GPU clean validation, full candidate, and one-shot
  promotion path have distinct identities and the same no-public-prediction,
  no-leaderboard-selection gates.
- A pre-launch automation audit also found that the v1 and v2 validation
  controllers would have downloaded separate kernel version-1 outputs into a
  shared local directory. Variant-specific output slugs now prevent that
  collision. The idle v2 controller was restarted to load the fix; no training
  or unrelated workload was interrupted.
- A fixed v1+v2 fusion path was committed before either member produced a
  selection result. It requires both independent 38,381,478-parameter members
  to pass their own sealed training audits and separate complete-movie clean
  validations; otherwise the ensemble stops without opening its validation.
  Eligible members are strict-loaded from distinct checkpoint hashes and
  averaged at the dense-logit and subvoxel-offset level, for a total of
  `76,762,956` learned parameters. It then receives a new 1/4/8-view selection
  and sealed acceptance evaluation rather than inheriting either member's
  score. The same projected-runtime and full-candidate gates apply before the
  separately named two-GPU candidate can submit. Member identities, hashes,
  counts, widths, depths, clean-validation hashes, and equal `0.5` weights are
  embedded in the private runtime; public predictions and leaderboard results
  remain outside the fusion path.
- The official CELLECT release was pinned at Git commit
  `3586070926f7f1fd5d8df37456861d22bdc63236` for fallback assessment. It
  includes a roughly 10.2 MB two-frame 3D U-Net checkpoint and two sub-1 MB
  matching heads under GPL-2.0, with explicit adjacent-frame embeddings,
  center/size/division outputs, and anisotropic `zratio=5` inference. The
  weights were trained for MSKCC confocal data and its preprocessing assumes a
  different TIFF layout and intensity/axis contract, so no quota is assigned
  to it until a hash-pinned Biohub adapter and clean frame-level smoke test
  exist. Its architecture remains a useful association/division fallback, not
  evidence that cross-domain checkpoints will improve this competition.
- A final-candidate audit found that the inherited public notebook labelled a
  local SciPy reimplementation as an official-metric proxy. It is close enough
  for diagnostics but is not the pinned patched organizer implementation, so
  it can no longer authorize a peak-rank submission. The frozen public control
  was recomputed with organizer commit
  `075fc5f5a52d11077f9dc2b074644618f26939e2`, patch commit
  `aa65e90aeb8a774ebb1b549e547787b87ac8a01c`, and scorer-lock SHA-256
  `1db65dee620059f19bf16633aa54a9f3379eb5d5bdff148a4037b949393b7a9c`.
  Its exact four-movie score is `0.9343483108193262`; the weakest exact movie
  scores are `0.8051376014` for `44b6_267148e4` and `0.8361469901` for
  `6bba_07e24132`, so pooled-only promotion would hide material failure modes.
- Every peak-rank full candidate now materializes its postprocessed integer
  validation graphs as `official_validator_candidate.csv`. After download, an
  isolated CPU environment verifies the scorer source/dependency lock, all
  frozen control and truth graph tree hashes, and the expected official control
  counts before scoring candidate and control side-by-side. Promotion requires
  at least `0.003` exact pooled score gain, at most `0.001` adjusted-edge
  regression, at most `0.005` regression on any complete movie, two-embryo
  reporting, and a non-replica result. The old public proxy remains recorded
  for diagnosis only. A control-as-candidate smoke test reproduced the exact
  baseline and was correctly rejected for zero gain and exact replication.
- A late public EDA isolated a z-axis localization floor on one hard movie:
  integer z errors can be as large as a cell's one-frame motion, while
  post-link coordinate smoothing cannot undo an earlier substitution. Source
  tracing showed that the active detector's continuous offsets survived into
  graph finishing but were rounded away before the frozen edge model. Before
  v1's first selection checkpoint, production association was precommitted to
  pass continuous detector coordinates into the linker's positional and
  pairwise-coordinate branches while explicitly rounding only feature-map
  lookup. This preserves the frozen model's trained feature contract, records
  `association_coordinate_mode=subvoxel` in every worker manifest, and remains
  subject to the unchanged exact complete-movie promotion gate. The public
  notebook's one-movie result and predictions are not used as score evidence.
- The newly released FOCUS-3D nuclei checkpoint was assessed as a possible
  independent heavyweight detector. Its public wrapper describes a roughly
  1.1B-parameter MaskFormer-style instance model and reports 98.5% sparse-node
  recall on one movie, but supplies neither a patched-official complete-movie
  score nor independent leaderboard evidence. The code is BSD-3-Clause; the
  official Hugging Face card declares no checkpoint license and is auto-gated,
  while the mirrored 4.5 GB Kaggle runtime uses the ambiguous `other` license.
  Antelume currently has only 2.1 GB free. The lane is therefore held without
  downloading weights, consuming GPU, copying predictions, or disturbing the
  active queue; it can reopen only after explicit weight licensing and safe
  storage are available.
- Executed output from the advertised `biohub-detfusion-sdw30-exact0943`
  notebook was audited before the active detector reached its first selection
  checkpoint. The notebook explicitly ran with its validator disabled and zero
  held-out samples. Its emitted guard report is stale: it records secondary
  detector weight `0.475`, threshold `0.96875`, and gap distance `5.8`, while
  the executed configuration used `0.30`, `0.965`, and `5.0`; it also declares
  leaderboard feedback use and `candidate_unverified` quality. Consequently,
  neither the claimed `.943` parent nor the new confidence-dominance repair is
  admitted to candidate selection. Our first candidate remains an independent
  detector test on the frozen linker and patched complete-movie scorer.
- NucVerse3D was reopened after a selective-download route removed the earlier
  7.93 GB packaging obstacle. The official MIT source is pinned at commit
  `d809a2e6cf380342708b7a9107574b259e6b34eb`; the official Zenodo record is
  CC-BY-4.0. Only the generalized scaled checkpoint was range-extracted and
  verified (`486,294,864` bytes, SHA-256
  `1c4e288350b1a86d361359cdd02151744d418e9fcf7ebe588c1c77e2cd8bbd67`).
  Its exact 40,458,005-parameter Keras graph was exported to fixed-shape ONNX
  (SHA-256
  `ca16e1b26d21ae522d68aba384ee7121f5ba2de28a122c291d1e1e627601e871`),
  with TensorFlow-to-ONNX and ONNX-to-PyTorch maximum absolute errors below
  `3.4e-6` on controlled parity probes.
- A train-only physical-compatibility screen for that checkpoint is now queued
  behind all three peak-rank members on Antelume. It restores the fine Biohub
  physical scale by mapping a `16x32x32` crop to the model's published
  `64x128x128` field and uses the published foreground/gradient-attractor
  decoder constants. Stable optimization crops must pass foreground support,
  attractor recall, finite-output, and distance floors before the same frozen
  checkpoint may open the disjoint selection role. Both phases share a strict
  two-hour wall budget. Competition test data, public predictions, leaderboard
  feedback, and submission commands are absent.
- The first active peak-rank checkpoint at step 1,000 is not promotable.
  Synthetic held-out AP/recall are `0.978156/0.978248`, but real positive-only
  recall is `0.675325`, mean distance is `2.516388` voxels, and p90 distance is
  `6.0` voxels. This isolates real-domain localization as the present failure
  rather than insufficient synthetic capacity. Training continues to the
  precommitted later checkpoints; v2 removes sparse-real negative pressure and
  v3 adds capacity only after v2.
- A same-day public refresh found no complete, auditable new submission model.
  The advertised Greenfield seed-A scorer describes a strong 13-epoch,
  124-train/32-development/39-holdout contract but is cancelled with no output
  and no publicly attached exact checkpoint. The SAM4CellTracking run is also
  cancelled, uses public `.943` detector nodes, and has no submission output.
  The new Xiaolei two-U-Net fork is code-contained in an earlier public
  notebook and reports only `0.5731/0.6589` checkpoint validation recall.
- The useful new evidence is diagnostic: one complete EDA attributes most
  remaining errors on `6bba_05db0fb1` to transiently faint cells that receive
  no on-cell detection, and a current third-place forum response recommends
  improving detection before linking and divisions. A project-authored
  temporal fading augmentation is therefore the next independent training
  member; it will not encode a movie identity, copy public predictions, or use
  a leaderboard-selected constant.
- The temporal-fading hypothesis is now an executable fourth peak-rank member.
  It uses the 66,977,670-parameter capacity architecture, seed `3601079`, and
  the same 2,000-step/7-hour bounds as v3. Generic soft local attenuation is
  applied to a random subset of labeled training points on the center frame,
  sometimes extending to one adjacent frame; probabilities are `0.70` for
  complete synthetic examples and `0.40` for sparse-real positives. Axial
  attenuation and conservative positive-unlabeled real loss are retained.
  The augmentation has no movie-specific constant and changes neither point
  coordinates nor labels.
- V4 is queued behind the gated NucVerse optimization/selection screen. It may
  reclaim only the exact v3 and NucVerse Biohub copies after independent local
  harvesters verify their archives and acknowledge the archive hashes back to
  Antelume. The capacity-v3 harvester was extended with that acknowledgement;
  a new NucVerse verifier checks archive paths, every member hash, phase-access
  flags, and the optimization-before-selection boundary. Eighteen focused
  tests plus Python/Bash/PowerShell syntax validation pass.
- V4 now has a complete autonomous promotion lane rather than ending at a
  checkpoint archive. Distinct builders package only its verified 67.0M
  checkpoint, create an offline private two-T4 complete-movie validation
  kernel, and build a separately named tracking candidate. The generic
  candidate controller still requires the exact patched scorer gain and
  per-movie regression gates before its one permitted submission. Both v4
  controllers pass validation-only preflight and wait on the v4 harvest.
- A capacity-diversity ensemble is precommitted without adding another AWS
  training run. It requires both 66,977,670-parameter v3 and v4 members to pass
  their own sealed audits and separate clean validations, then averages dense
  logits and offsets with fixed equal weights. The resulting 133,955,340-
  parameter v5 ensemble receives a new complete-movie validation, TTA/runtime
  selection, patched-official candidate score, and one-shot submission gate.
  Fourteen focused ensemble/controller tests and both controller preflights
  pass; failure of either member keeps the ensemble closed.
- A complementary v6 fusion is frozen before v3/v4 results. It takes the
  larger member logit at each voxel and the offset from that winning member,
  allowing the faint-cell specialist to contribute local peaks without being
  averaged away. It reuses only independently clean-promoted v3/v4 checkpoints
  and must pass a distinct complete-movie, runtime, patched-score,
  non-replica, and one-shot submission chain. It adds no AWS training cost.
- The training-only competition replay was expanded before any downstream v7
  result was observed. The same 96 optimization movies now contribute five
  temporally separated annotated frames each, increasing the optimization role
  from 146 to 480 frames and 3,592 annotated cells. The frozen 17-frame
  selection and 14-frame sealed audit roles are byte-identical, and the four
  final probes remain excluded. The inventory SHA-256 is
  `a80c9028b21fdba1746bc2686dc5c64f0852e7954d1e758ba1357bcca5aa0edc`.
- The expanded-real v7 member is queued behind v4 on the shared A10G. It uses
  the independently authored 66,977,670-parameter capacity model, conservative
  sparse-positive loss, depth attenuation, temporal fading, seed `4709011`,
  3,000 steps, and a 36,000-second AWS wall guard. The longer schedule was
  frozen before deployment after v1 measured about 7.6 steps/minute; validation
  still checkpoints at steps 1,000, 2,000, and 3,000 so an earlier generalizing
  state can win. Deployment waits for local
  hash-verified receipts for both the v4 archive and Kaggle CPU-expanded replay;
  the GPU yield guard pauses only named Biohub training if an unrelated GPU
  client appears. A distinct harvest, two-T4 clean validation, patched official
  metric comparison, and conditional one-shot submission chain are active.
- A second September 7 executed-output audit rejected every new public shortcut.
  The advertised public `0.942` notebook says detector threshold `0.96` was
  selected through leaderboard submissions and is `0.964152` line-overlap with
  `948tta2`. The latter's downloaded four-movie validator aggregates to only
  `0.934940` before the project's stricter exact adjustment. Greenfield's
  structurally independent 124-movie/13-epoch run was cancelled after one
  15-movie cross-fit direction and has no complete receipt; SAM4CellTracking
  was cancelled during inference, emitted no submission, and consumes public
  `.943` detector nodes. None enters model selection or the ensemble.
- A fixed three-member v8 ensemble was precommitted before v3, v4, or v7
  produced clean-validation results. It averages dense logits and offsets from
  those three 66,977,670-parameter models with equal `1/3` weights, for
  200,933,010 learned parameters. All three members must independently pass
  their sealed training audit and complete-movie clean validation; otherwise
  the v8 controller exits without building a runtime. An eligible ensemble is
  revalidated end to end, including runtime projection and the patched exact
  score/per-movie gates, before its distinct one-shot candidate may submit.
- A read-only AWS queue audit at `2026-09-07T07:22:18Z` found v1 healthy at
  step 1,750 with the A10G at 100% utilization. V2, v3, NucVerse, and v4 were
  all waiting in sequence; v3 used the shorter `bash ./run.sh` process name and
  was initially absent from a narrow display filter, but its held `run.lock`
  and PID `71141` proved it was active. A duplicate start was rejected by that
  lock, so no workload was interrupted or reordered.
- Two diverse public TemporalUNet checkpoints were screened as possible
  training-only teachers against the frozen real selection crops. Their
  top-64 localization recall was only `0.564935` and `0.623377`, their average
  was `0.584416`, and all p90 nearest-peak errors exceeded `10` voxels. High
  on-cell probability coexisted with many saturated false local maxima, so the
  distillation hypothesis was rejected without consuming GPU or admitting any
  public weight/logit into training.
- A ninth independent detector member is queued behind v7 on Antelume. It
  retains the 66,977,670-parameter capacity architecture, expanded 480-frame
  real optimization role, conservative positive-unlabeled policy, temporal
  fading, seed `5803219`, 3,000 steps, and 36,000-second wall guard. Its only
  change is a train-only local-shape margin: each annotated center is ranked
  above a fixed two-voxel shell, while shell locations near any other known
  center are excluded. This teaches a unique local maximum without assigning
  background labels across an incomplete real crop. The hash-pinned waiting
  wrapper is remote as PID `140809`; local harvest, dual-GPU complete-movie
  validation, patched exact scoring, and conditional one-shot submission
  controllers are active. The queue still yields to unrelated GPU clients and
  contains no RSNA process or path operation.
- A fixed v10 ensemble is precommitted before either expanded-real member has
  produced a result. It averages dense logits and offsets from v7 and v9 with
  equal weights (133,955,340 parameters total), and can materialize only if
  both members independently pass their sealed audits and separate clean
  complete-movie validations. The ensemble then receives its own two-GPU
  validation, runtime projection, patched exact score, per-movie regression
  gate, non-replica check, and conditional one-shot submission. It adds no AWS
  training cost and cannot conceal a rejected member.
- A train-role-only localization diagnostic on the frozen expanded-real crops
  tested fixed blob priors without opening sealed, test, or leaderboard data.
  The existing Gaussian difference-of-Gaussians reached `0.766234` top-k
  recall with mean/p90 nearest-center distances `2.1211/5.5893` voxels. A
  uniform `avg3-avg9` bandpass improved recall to `0.785714` and distances to
  `2.0223/4.9823`; raw intensity reached only `0.538961`. This supports an
  architecture change rather than a leaderboard-selected threshold.
- V11 is a 66,984,582-parameter blob-aware detector queued after v9 on the
  shared Antelume A10G. It preserves v9's expanded-real, temporal-fading, and
  local-shape training contract, while adding fixed `avg3-avg9` local-contrast
  channels for the current frame and temporal mean to the learned stem. Its
  distinct model-family tag is required by harvest, runtime packaging, strict
  checkpoint reconstruction, and inference. The deployed 3,000-step run uses
  seed `6902243` and a 36,000-second wall guard; it remains compatible with the
  Biohub-only GPU yield mechanism and does not address RSNA paths or processes.
- A fixed v12 ensemble was precommitted before v7, v9, or v11 produced clean
  validation results. It averages dense logits and offsets from all three
  independently gated expanded-real members, totaling 200,939,922 learned
  parameters. Mixed standard/blob strict loading has a forward-pass test. V12
  can materialize only after all three individual complete-movie validations
  pass, then receives its own offline private two-T4 validation, runtime check,
  patched exact and per-movie gates, non-replica audit, and conditional one-shot
  submission. Its background validation and candidate controllers are active.
- The active v1 detector reached step 2,300. At the frozen step-2,000 check it
  still rejected cleanly: synthetic AP/recall were `0.985361/0.985526`, while
  real positive-only recall remained `0.688312`, mean distance was `2.465151`,
  and p90 distance was `6.0` voxels. This confirms that synthetic saturation
  is not a reason to add only generic convolutional capacity.
- V13 adds global context to the strongest independent detection hypothesis.
  It is an 83,788,422-parameter blob-aware temporal ConvNeXt U-Net with two
  pre-normalized full-field attention/MLP blocks at the `8x8x8` bottleneck.
  The design follows the general local-plus-global multiscale finding reported
  by SwinCell (<https://www.nature.com/articles/s42003-025-08397-x>) without
  copying its code or weights. Training retains expanded real coverage,
  positive-unlabeled treatment, temporal fading, local-shape margin, and seed
  `7013267`; a 43,200-second wall guard bounds the heavier 3,000-step run.
- Remote v13 deployment initially stopped during import-only preflight because
  the staged global module coupled base-model and blob-module fallback imports.
  No runner or GPU process started. The imports were separated, all hashes were
  regenerated, the exact lock-free partial v13 directory was removed, and the
  corrected hash-bound runner was deployed behind verified v11. It uses the
  Biohub-only yield-compatible executable name and contains no operation on an
  RSNA path or process.
- V14 is a precommitted 150,773,004-parameter equal-logit/offset ensemble of
  v11 and v13. Both members must first pass their individual sealed audits and
  complete-movie validations. V14 then receives its own offline dual-T4
  validation, runtime projection, patched exact and per-movie gates,
  non-replica audit, and conditional one-shot submission.
- Kaggle GPU launches now use an account-wide named mutex across every peak-
  rank validation and candidate controller. Immediately before each push, the
  controller parses live `kaggle quota --format json` and rejects the launch if
  the reported remaining hours minus the declared 12-hour worst case would
  fall below the eight-hour reserve. All idle validation/candidate waiters were
  restarted with this gate; AWS harvesters and training processes were not
  stopped. The refresh-time balance was 30.00 hours remaining.
- A same-time authenticated public refresh found no notebook newer than the
  already rejected metadata-only planner. External repository
  `matt-ceran/biohub-cell-tracking` at commit `446589b7` reports a clean DoG plus
  small positive-unlabeled CNN detector, but only `0.674186` mean local edge
  Jaccard, no released checkpoint, failed division policies, and no repository
  license. It supplies corroborating generic blob-detection evidence only; no
  code, weight, prediction, constant, or score is admitted.
- A fourth authenticated September 7 notebook refresh at approximately
  `09:13Z` was unchanged: the newest competition notebook remained the
  metadata-only Zarr memory planner from `07:28Z`. No new runnable model,
  checkpoint, clean complete-movie validation, or submission artifact was
  available, so the public audit changed no experiment or threshold.
- The active v1 AWS run reached step 2,600 with the A10G at 100% utilization.
  Its latest completed gate remains step 2,000: synthetic AP/recall are
  `0.985361/0.985526`, but sparse-real positive recall is only `0.688312`
  with mean/p90 distances `2.465151/6.0`. This gap rejects additional generic
  convolutional depth as the sole next change and prioritizes real-domain
  scale and false-peak handling. The read-only inspection did not signal,
  pause, or modify any GPU client.
- A reproducible diagnostic scanned all 480 expanded-real optimization crops
  (3,592 annotated center-frame points) directly from the SHA-256-pinned
  competition-train replay archive. It did not open selection, sealed audit,
  competition test, or leaderboard artifacts. With the same top-64 local-peak
  contract, the current `avg3-avg9` response recalled `0.438474`; the wider
  `avg5-avg13` response reached `0.484410`, a `+0.045935` absolute gain. A
  predeclared equal-standardized `avg5-avg13 + avg7-avg15` fusion reached
  `0.488864`. Absolute numbers are diagnostic rather than promotion evidence,
  but they establish useful scale complementarity on the optimization role.
- V15 converts that training-role evidence into an independently trained
  83,802,246-parameter detector. It retains v13's two global bottleneck blocks
  and expanded-real/faint/local-shape contract, while its stem receives
  current-frame and temporal-mean Difference-of-Averages channels at fixed
  `(3,9)`, `(5,13)`, and `(7,15)` scales. The box filters are exactly separable
  to avoid the cubic cost of 13- and 15-voxel kernels. Seed `8124071`, 3,000
  steps, real frequency two, and a 50,400-second AWS wall guard were frozen
  before deployment. The hash-bound remote runner is PID `180979`, waits for
  verified v13 harvest, yields to unrelated GPU clients, and contains no RSNA
  path or process operation.
- V15 has an autonomous hash-verified harvest, private dual-T4 complete-movie
  validation, patched-official metric/per-movie promotion, non-replica audit,
  and conditional one-shot candidate chain. Kaggle launches share the existing
  account-wide mutex and live 8-hour reserve gate. No submission has been made.
- V16 is a precommitted 234,575,250-parameter equal-logit/offset ensemble of
  v11, v13, and v15. It can materialize only if all three members independently
  pass sealed training audit and separate clean complete-movie validation.
  It then receives its own dual-T4 runtime projection, patched exact metric,
  per-movie regression, non-replica, and one-shot submission gates; failure or
  runtime excess of any member or the triad closes the lane.
- The optimization-only scale diagnostic was extended to quantify whether a
  generic intensity boundary can safely identify negative real voxels. Across
  the same 480 crops and 3,592 annotated points, the `avg5-avg13` value at an
  annotated center has a median of `+17.537430` robust background scales and a
  tenth percentile of `+1.639818`; only `0.028118` of annotated centers fall
  at or below the per-crop response median. For `avg7-avg15`, the corresponding
  below-median fraction is `0.055958`. These are optimization-role diagnostics,
  not validation or leaderboard metrics.
- V17 uses the lower-risk `avg5-avg13` evidence to suppress a failure mode that
  conservative positive-only real loss cannot address. On sparse-real steps,
  each known center is ranked above the 16 hardest candidates in a 3-to-8-voxel
  shell only when their fixed blob response is at or below the crop median;
  candidates within 2.5 voxels of any known center are excluded. Bright
  unlabeled cell-like structures remain unlabeled and receive no negative
  loss. The new term has weight `0.25` and margin `0.5`; complete synthetic
  supervision is unchanged.
- V17 retains the 83,802,246-parameter v15 multiscale/global architecture,
  expanded-real coverage, temporal fading, and local-shape objective, but uses
  independent seed `9235183`. Its 3,000-step run has a 50,400-second AWS wall
  guard. The hash-bound runner was deployed at `2026-09-07T10:08:18Z` as PID
  `192045`, waits for verified v15 harvest, yields to unrelated GPU clients,
  and contains no operation on an RSNA path or process.
- V17 has a separate hash-verified harvest, private dual-T4 complete-movie
  validation, patched exact metric/per-movie promotion, non-replica audit, and
  conditional one-shot submission chain. The fixed evidence band, median
  boundary, shell radii, and `0.028118` optimization diagnostic are embedded in
  and checked against its terminal receipt.
- V18 is a precommitted 167,604,492-parameter equal-logit/offset ensemble of
  v15 and v17. Both members must pass their sealed training audits and separate
  clean complete-movie validations before the pair can materialize. It then
  receives its own dual-T4 runtime projection, patched official score,
  per-movie regression, non-replica, and one-shot submission gates. No Kaggle
  job or submission was launched while defining this lane.
- The original 38,381,478-parameter v1 run completed all 3,000 steps and was
  rejected at its frozen real-selection gate, so neither sealed audit nor
  Kaggle validation opened. Its last checkpoint improved real positive recall
  from `0.688312` at step 2,000 to `0.811688` at step 3,000, while mean/p90
  localization distance improved from `2.465151/6.0` to `1.898394/5.896`.
  This misses the predeclared `0.85` recall and `3.5`-voxel p90 requirements,
  but the late improvement supports a longer schedule for a stronger model.
  Per-embryo analysis exposed the dominant imbalance: 44b6 had five annotated
  points across three selection crops and only `0.20` weighted recall, versus
  `0.832215` recall across 149 points for 6bba. The optimization inventory is
  correspondingly uneven at 150 versus 330 crops.
- V19 is a predeclared 83,802,246-parameter continuation of the v17
  evidence-filtered safe-rank architecture. It duplicates each 44b6
  optimization crop once, changing effective optimization sampling from
  `150:330` to `300:330`; the 17-frame selection role and 14-frame sealed audit
  remain unchanged and are not duplicated. The independent seed is `10346297`,
  the schedule is 5,000 steps with checks every 1,000, and the AWS wall guard
  is 72,000 seconds. This tests the two concrete v1 findings—embryo imbalance
  and continued late learning—without using test, leaderboard, or held-out
  labels to tune a threshold. Its hash-bound remote runner was deployed as PID
  `204156`; it waits for the exact v17 archive to be locally verified and for
  all GPU clients to exit before training, and removes only the acknowledged
  v17 result copy. Harvest, private two-T4 validation, and one-shot candidate
  controllers are active with 240-hour wait windows.
- V20 is precommitted before any v17 or v19 result. It equally averages dense
  logits and offsets from the independently seeded safe-rank and balanced
  members, totaling 167,604,492 parameters. Both members must independently
  pass sealed training audit and clean complete-movie validation; V20 then
  receives its own private dual-T4 runtime, patched-official/per-movie,
  non-replica, and one-shot submission gates. A rejected member closes the
  ensemble rather than allowing it to conceal a failure. Its validation and
  candidate controllers are active, but cannot build or launch until both
  member validation receipts exist and are clean.
- The 10:45 UTC authenticated public refresh found two new notebooks, both
  copies of the already excluded public lineage. `test-biohub-run77` is
  byte-identical to its `biohub-run77` parent and has `0.964657` source overlap
  with `948tta2`; `biohub-lb942-fork-verbatim` is byte-identical to the
  previously audited `biohub-lb-942` and has `0.903958` overlap with `948tta2`.
  The discussion inventory supplies no new released, licensed, independently
  validated model. No public weight, prediction, constant, or score enters the
  v19/v20 decision.
- A fresh official-source check kept the roughly 1.1B-parameter FOCUS-3D
  checkpoint excluded. Its Hugging Face repository at commit
  `115258efcc9ee44e69db3902bce2511d0ae24e2f` is auto-gated, stores 13.4 GB,
  and still exposes no model-card license metadata. The BSD-3-Clause source
  license and CC-BY-4.0 paper do not establish a license for the separately
  distributed weights, while the Kaggle mirror continues to declare `other`.
- V21 implements the heavy-network hypothesis with owned, reproducible weights
  instead. It scales the v19 safe-rank/multiscale/global detector from
  83,802,246 to 129,748,646 parameters using widths
  `(160,320,640,1280)`, retains the 5,000-step embryo-balanced optimization
  contract, and uses independent seed `11457211`. Its 90,000-second A10G wall
  guard and hash-pinned runner are queued behind verified v19 as remote PID
  `214875`; it yields to every active GPU client and never addresses an RSNA
  path or process.
- V21's first deployment attempt failed during import-only preflight because
  the check imported an unstaged convenience function from `model.py`. No
  runner or GPU process started. The failed receipt was preserved, the exact
  lock-free partial v21 staging directory was removed, and the preflight was
  repaired to count parameters directly. The second deployment verified all
  staged hashes and the exact `129,748,646` count before starting its waiter.
- V22 is a precommitted 213,550,892-parameter equal-logit/offset ensemble of
  the independently seeded v19 and v21 balanced members. Both must pass their
  own sealed audits and clean complete-movie validations before v22 can build;
  the pair then receives its own two-T4 runtime, patched official/per-movie,
  non-replica, and conditional one-shot submission gates. Harvest, validation,
  and candidate controllers for v21/v22 are active with extended queue-safe
  wait windows.
- The raw competition movies have anisotropic voxels at
  `(1.625, 0.40625, 0.40625)` micrometres, but the frozen replay and inference
  contract reduce X/Y by exactly four. The detector therefore sees physically
  isotropic `(1.625, 1.625, 1.625)`-micrometre voxels; anisotropic kernels or
  strides would encode the wrong geometry and are rejected.
- An optimization-only raw/replay join tested whether phase-locked `::4` X/Y
  decimation was losing faint cells. The diagnostic hash-verified the expanded
  replay archive, reproduced each frozen decimated crop byte-for-byte, and
  evaluated 1,027 annotated points across 114 available raw crops without
  opening selection, sealed audit, competition test, or leaderboard artifacts.
  Current decimation achieved top-64 recall `0.395326` within 2.5 voxels. The
  best pooled alternatives, 50/50 and 75/25 mean/max blends, reached only
  `0.393379`; mean, RMS, and max were lower still. The predeclared `+0.02`
  promotion margin was missed, so anti-aliased replay rebuilding and a GPU
  pooling ablation are rejected.
- The `11:21Z` authenticated public refresh added no independent model.
  `rogerrogerroger3r/biohub-run79` (SHA-256 `5dc609c6...6eb727`) is a
  `0.995743`-overlap child of run77 that activates D4-averaged association
  features already represented in the frozen association bridge. Run80
  (SHA-256 `15ae7306...690554`) has `0.998576` overlap with run77 and changes
  only the leaderboard-selected detector threshold from `0.965` to `0.96`.
  Neither source, weight, prediction, score, nor threshold enters the project.
- The single-A10G dependency audit found ten serial, single-change ablations
  between the active v2 job and the balanced/XL candidates. V19 trains from an
  independent initialization and does not consume a predecessor checkpoint;
  its original v17 dependency was only a GPU-ownership ordering edge. To avoid
  days of low-priority queue latency, v19 was reproducibly reprioritized to
  start after v2's exact archive is locally harvested and acknowledged. The
  replacement script required an exact idle v19 waiter match, absence of a v19
  training log/results/archive, and changed only that waiter. The live v2 PID,
  NucVerse waiter, RSNA, and all unrelated processes were untouched. The
  installed runner hash is `57ef88c6...09d248`, replacement PID `225063`, and
  v21 remains queued behind verified v19.
- The priority deployment had two bookkeeping-only retries. The first failed
  its remote inspection before mutation; the second successfully installed and
  launched PID `225063` but its local parser included `sha256sum`'s `OK` line
  with the PID. Both failed receipts were preserved. A read-only recovery check
  verified the exact runner hash, sole idle waiter, and absent v19 training
  artifacts, then emitted the terminal priority receipt without restarting any
  process.
- The owned detector inference already applies D4 X/Y TTA to both dense logits
  and continuous offsets with the correct inverse vector basis. One symmetry
  used during training was missing from clean validation: Z reflection. Because
  the X/Y-pooled replay grid is physically isotropic, `zflip2` is now a frozen
  two-view option ordered between `none` and `rot4`. Calibration cost rises from
  13 to 15 aggregate views across the existing mode screen, rather than adding
  a prohibitive 16-view mode. The unchanged selector may choose `zflip2` only
  when it clears absolute selection floors and remains within `0.003` pooled
  and `0.01` worst-movie recall of the best calibration mode. Runtime manifests,
  promotion, production loading, and candidate verification all recognize the
  exact two-view contract; no leaderboard feedback selects it.
- The inference review also found a release-blocking validation defect:
  `evaluate_peak_rank_detector.py` called the metadata-only density threshold
  routine without importing it. Any real worker would have raised `NameError`
  before opening labels or scoring, despite mocked orchestration tests passing.
  Both packaged and repository import paths now bind the authoritative
  Spotiflow-bridge implementation, and a direct import regression test protects
  the worker. The expanded explicit peak-rank suite passes `206` tests.
- The `11:59Z` public refresh found one later notebook but no independent
  candidate. `rogerrogerroger3r/biohub-run81` has raw SHA-256
  `6cf32a96...4123953`, overlaps run79 at `0.998582`, and differs from it only
  by run80's public-leaderboard-selected detector threshold (`0.965` to
  `0.96`). It therefore stacks two already audited changes from the same public
  lineage rather than contributing a new architecture, trained member, or
  clean validation receipt; it remains excluded.
- Two trajectory resources mentioned in new discussion comments also fail the
  detector-ingress gate. Keller/SSBD contains quantitative tracks but no image
  datasets and is licensed `CC BY-NC-SA`. The Virtual Embryo Zoo zebrafish
  download is an Ultrack-derived tracking-only bundle with no linked image data
  and no explicit permissive dataset license; its related preprint is
  `CC BY-NC 4.0`. No archive was downloaded, and neither resource, coordinate,
  track, prediction, threshold, or score enters training or selection. The
  other refreshed discussion advice repeats already represented sparse-label,
  error-decomposition, tracklet, and instance-segmentation directions.
- The active depth-PU v2 detector reached its first precommitted selection gate
  at step 1,000. Synthetic performance is already strong (mean AP `0.977921`,
  mean recall `0.977982`, worst-movie AP `0.963479`), but real positive-only
  recall is only `0.357143`, mean distance is `4.093636` voxels, and all five
  annotated points in the three held-out 44b6 crops are missed. The gate is a
  clean rejection (`selection_passed=false`, composite `1.414692`), not a
  promotion candidate. A transient synthetic batch at step 1,050 produced loss
  `135.9` and pre-clipping gradient norm `25057`; the following real batch at
  step 1,100 returned to loss `0.062` and gradient norm `1.21`. The trainer
  already clips gradients to `5.0`, so the precommitted step-2,000 and
  step-3,000 gates remain useful and the live run is not interrupted. V19's
  balanced 44b6 sampling and safe-negative ranking remain the next independent
  test; no hyperparameter is selected from this held-out gate.
- An independent CellSeg3D SwinUNETR candidate was bounded on two
  optimization-only replay crops before any GPU allocation. The MIT-licensed
  model revision `09f946501deea177be59a7a2c4dc1c7ba36f3c94` supplied a
  `SwinUNetR_latest.tar.gz` archive of 270,715,626 bytes with SHA-256
  `d9729940...728f21`; the extracted checkpoint has SHA-256
  `9f73ae3a...821cea` and strict-loads into the documented 72,762,019-parameter
  MONAI SwinUNETR. On the 44b6 three-point crop, top-64 local-max recall is
  `0.0` and the best connected-component recall is `0.333333`. On the 6bba
  six-point crop, top-64 recall is `0.166667`; the best component result is
  `0.666667` only at threshold `0.6`, while component counts vary from 527 at
  `0.3` to 20 at `0.4`, 37 at `0.6`, and 43 at `0.7`. Pooled top-64 recall is
  `0.111111`, and the exploratory best component setting reaches only
  `0.444444` with unstable cross-crop behavior. This misses the detector
  compatibility floor, so CellSeg3D receives no AWS run and no selection,
  audit, test, or leaderboard artifact is opened.
- NISNet3D/NIS3D and deltaMic were also screened from primary sources without
  consuming GPU. NISNet3D's released source is `CC BY-NC-SA` and its referenced
  data archive exposes no compatible license; the separate NIS3D software is
  MIT but its 3.3 GB dataset record does not declare a usable data license.
  DeltaMic is permissively inspectable under `CC BY-SA 4.0`, but is an inverse
  mesh-rendering method with no released image-detector checkpoint. None enters
  training or candidate construction.
- The hash-locked multiscale optimization diagnostic now reports each embryo
  separately and passes all six focused tests. Across all 480 optimization
  crops, the best fixed two-scale response recalls `0.200422` of 474 annotated
  44b6 points versus `0.532713` of 3,118 annotated 6bba points. For the
  safe-negative `avg5-avg13` response, annotated-center standardized median is
  only `4.121372` in 44b6 versus `21.836735` in 6bba, while the fraction at or
  below the background median stays low in both (`0.018987` and `0.029506`).
  This identifies distractor rejection and embryo balance—not wholesale
  foreground absence—as the current training problem. It supports the already
  precommitted v19 policy that duplicates the 150 44b6 crops to 300 against 330
  6bba crops and applies evidence-safe negative ranking; no held-out role was
  opened and no new parameter was selected from selection feedback.
- The 12:30 UTC authenticated Kaggle refresh exposed three recently updated
  notebook entries, but all remain in the same public family. Current
  `biohub-harmonic-fusion` (SHA-256 `6e1f25c4...e985bb`) and
  `biohab-lineage-forge-adaptive-tracking` (`74c83cd0...9f35a`) have normalized
  source overlap `0.996589`; their substantive difference is the secondary
  link mode (`low_margin_consensus` versus `adaptive`). They overlap `948tta2`
  at `0.893394` and `0.892775`. The new 0.934-title notebook
  (`8853e3e1...b56aa0`) overlaps its author's 0.931 predecessor at `0.989019`
  and `948tta2` at `0.943435`; its visible changes are another mixture of
  detection, edge, bidirectional, gap, and division constants. All three
  explicitly set `leaderboard_feedback_used_for_configuration=True`, and their
  embedded terminal configurations retain stale values that contradict their
  live top-cell settings. They therefore fail clean-selection, independence,
  and reproducibility gates. Their D4 edge-feature TTA concept is already
  independently implemented in the owned association bridge; no public code,
  weight, prediction, threshold, or score enters the candidate.
- A second hash-locked optimization-only diagnostic screened eight fixed
  combinations of temporal reduction and ordinary versus local-variance-
  normalized multiscale blob evidence. The strongest existing two-scale prior
  has pooled recall `0.488864`, with `0.200422` on 44b6 and `0.532713` on
  6bba. Temporal-minimum local-SNR evidence reaches pooled recall `0.493318`,
  raises 44b6 to `0.236287`, and preserves 6bba at `0.532393`. Thus it is a
  Pareto-like improvement on the hard embryo rather than a pooled gain obtained
  by sacrificing the easier embryo. Mean and median local-SNR alternatives are
  slightly weaker, which keeps the follow-up to one fixed temporal-minimum
  construction. All 480 optimization crops are used; selection, sealed audit,
  competition test, and leaderboard artifacts remain closed.
- V23 is precommitted from that training-role evidence before v19 or v21 opens
  any selection result. It adds three temporal-minimum local-SNR bands to the
  v19 safe-rank/multiscale/global detector, retaining embryo-balanced sampling,
  5,000 steps, real frequency 2, widths `(128,256,512,1024)`, depths
  `(3,3,9,3)`, and the same clean loss contract, with independent seed
  `12568331`. The model has 83,812,614 parameters and a 72,000-second A10G
  guard. Its sequential runner is hash-pinned to every staged and repository
  dependency, waits for v21 completion plus verified local harvest, and checks
  for an idle GPU before launch. Focused diagnostic, model, trainer, evaluator,
  runner, and deployment tests pass; no public model or prediction is consumed.
- V23 was staged on Antelume with runner SHA-256 `7675d0f5...da2a4d5c`
  and entered its v21 harvest wait as PID 270072. The deployment preflight
  exposed a latent flat-import path omission shared by the queued multiscale
  models before any V23 GPU work began. Antelume now resolves the flat
  `model.py` import through a symlink to the same repository source whose
  SHA-256 (`da1eae6c...47c637`) is checked by every runner. The repaired
  preflight instantiated exactly 83,812,614 parameters; the active v2 detector
  remained the sole GPU process throughout. A hidden local controller (launch
  PID 7444) now waits to copy, independently verify, and acknowledge the V23
  archive. Its clean two-GPU validation and submission builders also pass
  preflight. The refreshed Kaggle GPU balance is 30 hours, leaving 22 spendable
  hours after the mandatory 8-hour reserve.
- V24 is precommitted, before either member exposes validation evidence, as the
  fixed equal-logit pair of v21 and v23. This combines the 129.7M-parameter XL
  capacity lane with the 83.8M-parameter temporal-minimum local-SNR lane for
  213,561,260 total parameters. The pair may be built only if both individual
  complete-movie validation controllers report clean promotion, and it must
  then pass its own complete-movie selection, sealed acceptance, patched
  official-metric comparison, worst-movie non-regression, runtime, and
  reproducibility gates. No validation result selects its members or fusion
  weight. The generic ensemble packager's architecture check was also repaired
  to recognize the already declared clean descriptive architecture strings for
  v19/v21/v23 while continuing to reject strings marked public.
- The local Kaggle launch queue contained 41 obsolete or dominated validation
  and candidate controllers, including duplicate v2 and superseded ensemble
  lanes. They had no terminal evidence and had not launched kernels; those
  controller processes were stopped without changing any training artifact or
  remote job. The live queue now retains v2, v19, v21 validation, v23
  validation, and the v24 validation/candidate path. Single-model v21/v23 and
  dominated v22 candidate launches are intentionally suppressed to preserve
  quota for the stronger v24 pair. Both launch controllers now release their
  account-wide GPU mutex and wait for quota refresh when the 12-hour declared
  budget would cross the 8-hour reserve, instead of permanently skipping a
  scientifically eligible candidate.
- The 13:50 UTC public refresh added an explanatory ILP audit, not a new
  candidate. Source SHA-256 `a8b20e81...5bfe1de` correctly derives that, with
  appearance cost zero, a native ILP fork needs the second-daughter edge
  probability to clear the division cost; the inherited `1.2` cost is therefore
  unreachable. The production trace adds two decisive facts: candidate edges
  are first cut at `0.48`, and each daughter can have only one parent. All four
  cached raw control graphs contain zero forks, while their postprocessed graphs
  contain `103/81/5/29`, confirming that the current division channel is the
  separately gated post-link rule.
- A CPU-only probe on one already-opened validation transition found the true
  parent's single-pass dual-linker probabilities for its two matched daughters
  were only `0.125664` and `0.242600`; the executed frozen graph assigned the
  second daughter to a competing parent at `0.647323`. This diagnostic cannot
  select a threshold, but it falsifies a cheap `division_weight < 1` candidate:
  changing that cost alone cannot restore a below-threshold edge or reassign an
  already claimed daughter. No active detector, candidate, controller, or GPU
  job was changed. Any future division experiment must explicitly learn or
  validate fork-aware reassignment under complete-movie edge and worst-movie
  gates.
- Discussion topics `739731`, `739685`, `739570`, `738276`, and `738778`
  reinforce synthetic-plus-real training, image-feature association,
  held-out-complete-movie validation, and high-precision rare-division handling.
  Topic `739018` proposes predicted-node-count adjustment and is excluded as a
  metric-hack direction; `739915` and `739516` provide no runnable evidence.
- Depth-PU v2 improved substantially at its precommitted step-2,000 gate but
  still failed cleanly. Synthetic mean AP/recall/worst-movie AP reached
  `0.984802/0.984996/0.968421`; real positive-only recall rose from the
  step-1,000 value `0.357143` to `0.629870`, and mean distance fell from
  `4.093636` to `2.792624` voxels. The frozen real floors remain `0.85` recall,
  `2.25` mean distance, and `3.5` p90 distance, and only one of five held-out
  44b6 points was recovered. The gate therefore remains
  `selection_passed=false`; the immutable step-3,000 gate continues, while the
  queued embryo-balanced detectors remain the justified follow-up.
- Depth-PU v2 completed its full 3,000-step, 23,166-second run and was
  hash-verified locally from archive SHA-256 `4ec7f060...c2209c`. Its final
  step regressed from the step-2,000 maximum to real recall `0.590909`, mean
  distance `3.027830`, and p90 distance `6.0`, despite synthetic mean
  AP/recall `0.986454/0.986646`. The final terminal is therefore a clean
  `rejected_at_selection` with audit unopened and no checkpoint authorized for
  Kaggle validation. The verified harvest acknowledgement releases the AWS
  queue to the embryo-balanced V19 follow-up; V2 will not be revisited.
- An optimization-only motion-supported local-SNR diagnostic (receipt SHA-256
  `fed17e3f...955d77f`) found no new detector member worth GPU time. The best
  motion-supported response raised hard-embryo 44b6 top-64 recall only from
  `0.236287` to `0.238397` while lowering pooled recall from `0.493318` to
  `0.481347`; the lane is rejected before selection. The diagnostic also found
  that staged V23 used `(3,9)/(5,13)/(7,15)` rather than the exact
  `(3,9)/(3,11)/(5,13)` scale set that justified it. Although the staged set
  raised pooled optimization recall to `0.506960`, it regressed 44b6 to
  `0.229958`, violating the precommitted embryo-balanced rationale. Because V23
  was still a dormant waiter with no training or held-out artifacts, its model
  and runner were reproducibly repaired to hashes `04f2c5d5...db172e` and
  `c5948e48...e88e7f`; the receipt (`0cd4eb73...a2cc2b`) records old PID
  `270072`, new waiting PID `310374`, and the exact scale contract. V2 remained
  the sole GPU process throughout.
- Public-teacher distillation remains closed: the two available TemporalUNet
  teachers were trained on all 199 competition movies, including the project's
  held-out stems, and their prior clean screen reached only `0.564935` and
  `0.623377` top-64 localization recall. Restricting new pseudo-label reads to
  optimization crops would not remove label information already encoded in
  those weights, so they are not admitted as clean promotion evidence and no
  additional GPU run was allocated to them.
- V27 converts the remaining real-domain failure into a precommitted,
  optimization-only hard-example sampler. The exact V23 temporal-minimum
  local-SNR response ranks all 480 hash-pinned optimization crops without
  opening selection, sealed audit, test, public predictions, or leaderboard
  data. The 50 hardest crops in each embryo have mean per-crop top-64 recall of
  `0.0` for 44b6 and `0.020527` for 6bba. Manifest SHA-256
  `9967efa2...dca900` balances the effective stream to `495:495`: 105/45 44b6
  crops receive multiplicity 3/4, while 165/165 6bba crops receive
  multiplicity 1/2. This changes training exposure, not validation membership.
- The 83,812,614-parameter V27 member retains V23's exact architecture,
  safe-rank loss, local-SNR bands `(3,9)/(3,11)/(5,13)`, and complete synthetic
  supervision, with independent seed `13679443`, 6,000 steps, real frequency
  two, and an 86,400-second wall guard. Its hash-verified Antelume waiter is PID
  `324895`, queued only after V23's locally verified harvest. At the deployment
  check V2 PID `189098` remained the sole GPU client; V19, V21, V23, and V27
  were sleeping in their declared order, and no unrelated process was touched.
- V28 is precommitted before V27 exposes any held-out result as a fixed
  213,561,260-parameter equal-logit/offset pair of the individually gated XL
  V21 and hard-mined V27 members. V27 standalone and V28 pair controllers may
  package, validate, and submit only after their immutable member evidence,
  complete-movie selection and acceptance, patched official score,
  worst-movie non-regression, runtime, non-replica, and live Kaggle quota gates
  pass. Their Kaggle launch paths retain the mandatory eight-hour reserve and
  two-GPU execution contract.
- The recovered graph-context division sweep isolates seed selection variance
  as a policy failure rather than evidence against graph context. All eight
  74,732,308-parameter members passed selection; the calibration-free
  equal-rank ensemble reached selection AP `0.951062` with 11 true positives
  before its first false positive. Selecting the apparent best individual
  (`0.992157` selection AP) did not generalize (`0.682187` audit AP), while two
  lower-ranked seeds passed the independent audit. Those audit outcomes are
  diagnostic only and cannot be used to retroactively choose either seed.
- Graph-context frozen ensemble v2 is therefore precommitted as a fresh-seed
  experiment, not a rescue of favorable audited members. It trains four new
  seeds (`1013131/1113137/1213139/1313141`) from each of the same two distinct
  backbone initializations, admits members on selection evidence only, and
  freezes every admitted member into one equal-within-movie-rank ensemble
  before audit opens. There is no strongest-seed fallback and no post-audit
  member removal. The audit gate evaluates that frozen ensemble as the policy
  unit; per-member audit outcomes remain diagnostics. If accepted, the same
  unit alone opens the still-sealed four-movie development probe. The declared
  run is eight sequential 74.7M models, 20,000 steps each, with a five-hour
  worst-case A10G budget after V27's verified harvest. Local trainer, probe,
  runner, deployment, and harvest preflights pass. AWS credentials were expired
  at the first deployment attempt, so hidden deployment PID `5564` now waits
  for refresh and hidden harvest PID `12444` waits for the hash-bound result;
  existing remote detector work is unaffected.
- The v2 downstream path preserves actual per-member audit outcomes rather
  than relabeling ensemble constituents as individually accepted. Runtime and
  notebook verification accept a failed constituent only when the immutable
  policy explicitly records `constituent_audit_gate_required=false`,
  `policy_unit_audited=true`, the v2 equal-rank contract, and at least two
  frozen members. Hidden PID `36076` will run the sealed development decision
  and package the private runtime after harvest. A briefly started graph-only
  candidate waiter (PID `19128`) was stopped before it launched or created a
  terminal because its inherited promotion gate uses an adjusted-edge proxy,
  not the patched official metric. The graph component may instead enter only
  the precommitted V30 composition, whose patched official score must beat the
  independently promoted V28 detector candidate.
- V30 is frozen before either V28 or graph-context v2 exposes candidate
  evidence. It composes the unchanged 213.6M V28 equal-logit detector with the
  unchanged graph-context ensemble and independent morphology voter; there is
  no learned fusion weight, member search, absolute division threshold, or
  leaderboard selection. Graph models load only after production detector
  inference, park on CPU during the two-worker held-out detector pass, and
  return to their fixed GPU partitions for complete-movie graph scoring. This
  avoids contaminating the detector runtime or creating a predictable VRAM
  collision. Promotion requires V28's independent patched-official promotion,
  graph v2's audit plus sealed-development acceptance, a new patched-official
  complete-movie score gain of at least `1e-6` over V28, and worst-movie
  regression no worse than `0.001`. The V30 controller retains the global
  Kaggle GPU mutex, two-T4/no-Internet contract, 12-hour ceiling, and eight-hour
  reserve; only the external submitter can submit after every gate passes.
  Hidden controller PID `31084` now waits for both independent component
  terminals and will execute that path without manual intervention.
- Antelume remained directly reachable over its private host while the local
  AWS session was absent. V19 was confirmed active on the A10G at 9,580 MiB;
  no RSNA process was displaced. The graph v2 deployer now supports that
  already-authorized direct route, and the immutable graph runner was queued
  as remote PID `354579`. It still cannot enter the GPU until V27 is complete,
  hash-verified locally, acknowledged remotely, and the accelerator is idle.
  This removes credential refresh from the queue's critical path without
  relaxing sequential ownership or evidence gates.
- V31 is a precommitted conditional capacity test, not another speculative
  queue member. It combines the exact V21 XL widths `(160,320,640,1280)` with
  the exact V27 temporal-minimum local-SNR architecture and frozen
  optimization-only hard-example sampler, yielding 129,761,606 parameters at
  a fresh seed for 6,000 steps. Its local controller cannot deploy unless V21
  and V27 each independently finish clean validation with promotion enabled
  and graph v2 is hash-verified and acknowledged. Parent rejection produces a
  terminal skip with zero AWS GPU use. If admitted, the remote runner still
  waits for graph completion and GPU idleness, has a 108,000-second watchdog,
  and its downstream dual-T4 validation/candidate controllers retain the
  eight-hour Kaggle reserve and external-promotion submission boundary.
  Hidden conditional controller PID `42212` is now waiting on those three
  immutable evidence terminals.
- A live process audit corrected the assumed queue owner: capacity-PU V3 won
  the post-V2 idle race and was using 9,580 MiB on the A10G, while the first
  priority V19 waiter had exited because V3 reclaimed V2's remote archive
  before V19 could independently re-check it. The active V3 run was preserved.
  V19 was rebound to V3's locally verified archive/ack, and repaired remote
  waiter PID `363500` now makes it the next model owner. It still yields to all
  active GPU processes and removes only the verified V3 Biohub artifacts.
  Superseded V4-V17 waiters do not have a live path into the GPU; the intended
  effective chain is now V3 -> V19 -> V21 -> V23 -> V27 -> graph v2.
- The 2026-09-07 public refresh found a newly popular notebook,
  `kunaldesale2408/biohub-cell-tracking` (download SHA-256
  `6dc327ce...f67568`). Static inspection classifies it as another derivative
  of the public dual-seed harmonic TemporalUNet/transformer/ILP stack, not an
  independently validated new model family. Its material changes are mostly
  leaderboard-described post-processing knobs (including detection `0.960`,
  division weight `1.3`, minimum track length `5`, wider divergence, and
  rescue thresholds), and its source explicitly labels the axis as targeted
  improvement over an LB score. It contains no execution outputs or raw
  prediction archive from which those knobs could be evaluated cleanly.
  Therefore neither the notebook nor its parameter bundle is admitted as
  promotion evidence; isolated ideas may be reconsidered only as
  precommitted hypotheses with complete-movie patched-official validation.
- The same discussion refresh reinforces the current experiment ordering.
  In `what layer did ur gains actually come from` (discussion 737543), a
  reported top competitor recommends solving detection before linking and
  division; other current threads separately focus on 3D detection, GNN
  association, and sparse-GT overfitting. Host discussions 727154 and 728324
  confirm the patched scorer/rescore, while discussion 728613 identifies
  nonconsecutive lineage edges as a catastrophic structural failure. No
  leaderboard-only score was used for model selection. The committed queue
  remains detection-first, and V30 now fails closed if its final ranked graph
  contains a dangling or nonconsecutive edge before `submission.csv` can be
  accepted.
