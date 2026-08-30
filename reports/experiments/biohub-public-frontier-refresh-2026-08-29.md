# Biohub public frontier refresh — 2026-08-29

Status: authenticated read-only source, output, rules, leaderboard, and
discussion audit completed. The active score target is `0.945`. Public
predictions are controls only and are not eligible submission artifacts.

## Public lineage

| Public kernel | Source SHA-256 | Observed result | Lineage finding |
|---|---|---:|---|
| `grafael/biohub-ct-0940-ema` | `4f0fb3aff772f4aaa0b687477e255c05211412a458d9db085935a7f37f073f33` | displayed `0.940` | `0.99238` normalized line-set Jaccard with the public 0.938 notebook; the substantive addition is a four-frame EMA velocity state with `alpha=0.4`, plus restoration of the older secondary detector blend |
| `grafael/biohub-ct-0938-harmonic` | `6d0f4314afdb1f90021898edb3559841152e21c4e3b197dcae460d368a52aedd` | displayed `0.938` | `0.99652` normalized line-set Jaccard with the public 0.927 lineage; changes are chiefly bidirectional and secondary-detection scalars |
| `notoverkil/biohub-base-0937` | `eb01b88a7128ff167cf73f3b0d1d8fc77f38618a0c9a94de1e7b0d886acf0a30` | displayed `0.937` | public-derived detector/linker with widened safe-division rules and the same four-movie validator family |

The 0.940 notebook is therefore a strong public control, not a new heavy model
family. Submitting it unchanged would be an exact public replica and is
excluded. Its EMA state is a useful hypothesis that may be independently
reimplemented and validated.

## Complete-movie validation audit

The downloaded outputs include the same four held-out complete movies. The
table below recomputes pooled edge and division counts from each
`validator_results.csv`; no threshold was selected from leaderboard feedback.

| Control | Adjusted edge Jaccard | Edge TP / FP / FN | Division TP / FP / FN |
|---|---:|---:|---:|
| public 0.937 | `0.915784` | `2187 / 109 / 106` | `1 / 1 / 4` |
| public 0.938 | `0.916462` | `2186 / 110 / 107` | `1 / 3 / 4` |
| public 0.940 EMA | **`0.917882`** | **`2188 / 103 / 105`** | `1 / 2 / 4` |

The EMA control recovers two more true edges than 0.938 while removing seven
false edges. Its division recall remains only `1/5`. The four-movie sample is
too sparse to select division thresholds reliably, but it identifies the
highest-value lane: recover complete division events without introducing
false forks. One additional true division can move the composite much more
than a small generic edge-linking adjustment.

An oracle feasibility audit of the 0.940 kernel's raw prediction GEFFs adds a
more specific diagnosis. All five annotated division triplets have the parent
and both daughters detected within 7 µm. Four triplets are fully matched within
3 µm; the remaining event has only its parent within 3 µm and daughters about
4.7–5.0 µm from truth. Every raw graph already connects exactly one correct
daughter. In all five cases the omitted daughter is the nearest next-frame
detection to the parent after excluding the existing child. Three omitted
daughters are parent-free; two require replacing an incoming edge. Adding the
five known missing edges as an oracle changes raw edge TP/FN by `+5/-5` and
raw division TP/FN from `0/5` to `5/0` without adding an oracle FP.

This result is feasibility evidence only: truth identified the missing
daughters, so it cannot authorize a distance threshold or a submission. It
does establish that a learned division-parent gate plus constrained
nearest-second-daughter selection has enough candidate recall to close the
target gap. The reproducible audit implementation is
`research/division_recovery_feasibility.py`; its ignored run artifact is
`.biohub/cache/analysis/division-recovery-0940.json`.

## Rules and discussion evidence

- External public data and models are allowed when competitors can obtain them
  with minimal cost; ZebraHub use is organizer-confirmed and declared to have
  no test overlap.
- Winning code must be reproducible and MIT-licensed. The competition data is
  CC0. Notebook submissions are limited to five per day and two final choices.
- Train and hidden test are embryo-disjoint, and the hidden test input is
  swapped into a rerun. Validation must therefore be complete-movie and
  embryo-aware.
- Discussion 737543 reports the common failure mode of very high detection
  recall but poor division Jaccard, and notes that useful division localization
  is substantially tighter than the nominal node-matching radius.
- Discussion 737101 recommends decomposing missing edges into missing endpoints
  versus incorrect associations. Discussion 734604 recommends measuring node
  recall, conditional linking, division candidate recall, and division ranking
  separately.

References:

- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737543>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737101>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604>
- <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330>

## Active candidate decision

The project-authored v4 external pretraining run uses two T4 GPUs, 46,386,607
parameters per fold, ZSNS004 optimization, and disjoint ZSNS005 selection and
audit windows. It reads no competition data. A recovery controller will verify
both checkpoints and will not launch the ceilinged competition-transfer stage.

The 47,964,082-parameter division localizer now warm-starts directly from that
verified external v4 output. It trains only the independently designed
localization objective, freezes the selection checkpoint before audit, and
changes coordinates only for predicted division parents and their immediate
daughters. The runtime is private and the training notebook has no competition
input or submission command.

To meet `0.945`, localization alone is not assumed sufficient. The candidate
must combine the 0.940 control's strongest reproducible elements with at least
one project-authored, validation-positive component:

1. learned division-event localization and division ranking from the v4 lane;
2. an independently evaluated robust motion state beyond fixed EMA; or
3. both, if each passes its frozen complete-movie gate.

No candidate is authorized for submission merely because training completes.
It must preserve the clean public control's edge quality, improve the frozen
complete-movie composite, remain non-identical to every audited public output,
finish inside the notebook limit, and pass strict output verification.

## Frozen `0.945` promotion path

The first submission candidate is now specified as the attributed clean public
control plus an independently reproduced four-frame velocity EMA and the
project-authored external division gate. The public safe-division heuristic is
disabled. The learned stage may add only the nearest parent-free second
daughter after an existing child; it cannot reassign an edge or modify a node
or coordinate. Every model checkpoint, policy, runtime source, public-control
notebook, and validator control is SHA-256 bound.

Policy calibration has two equivalent execution routes. Antelume uses one A10G
in the isolated `/home/ubuntu/biohub*` workspace. A Kaggle fallback attaches
only the private source runtime, the external ZebraHub shards, and the verified
v4 pretraining output; it attaches no competition source and has no submission
path. Both routes freeze the threshold on ZSNS005 selection windows and open
the disjoint audit windows once.

The external promotion gate compares the candidate with the exact public 0.940
validator artifact (`4dbf2079...e4941f4b`). It requires all of the following:

- candidate output SHA-256 differs from all three audited public submissions;
- at least one learned edge is added, with zero reassignments and zero
  node/coordinate changes;
- complete-movie proxy gain is at least `+0.005` over the public control's
  `0.932168` proxy;
- adjusted edge Jaccard regresses by no more than `0.001`;
- division TP and division Jaccard both improve;
- the two-T4 notebook completes before the 3,000-second watchdog boundary.

Only a hash-bound output passing every condition is submitted. An exactly-once
receipt is written before the external submit command, preventing a controller
restart from producing a duplicate submission.

## Contextual-v3 division-head control

While v4 was training, the two verified contextual-v3 folds were evaluated as
a diagnostic-only control with the same nearest-second-child policy. The
threshold selected on the external ZSNS005 selection windows made `2/2`
correct decisions. Applied unchanged to the disjoint audit windows, it made
`2/4` correct decisions (`0.50` precision, `2` division TP). This is positive
separability evidence, but the precision is below the v4 authorization gate and
the v3 audit had already been opened by its original training run. It cannot
authorize competition evaluation or submission.

The result strengthens the rationale for the v4 objective: useful division
signal already exists in the smaller model, while false-positive suppression
is the remaining bottleneck. The immutable diagnostic artifact is
`.biohub/staging/contextual-v3-division-recovery-diagnostic-v1.json` with
SHA-256 `a149ed7d6d3849bc2cb88159c5e4592d0466287b476e6b1029fb7aad69a32b51`.
The run also exposed and repaired a missing SciPy dependency in the isolated
Antelume Biohub environment before the v4 policy calibration needs it.

## V4 and focused-division outcome

The two-T4 v4 pretraining notebook completed 4,000 optimizer steps in both
folds, but its frozen association gate correctly rejected both checkpoints:
each best checkpoint remained at step zero and numerically identical to the
contextual-v3 initialization. The recovered 46,386,607-parameter checkpoints
were therefore allowed only as prediction-preserving initialization for a
division-specific experiment; they were never promoted as association models.

Two focused Antelume A10G runs optimized 25,178,047 division-relevant
parameters while freezing 21,208,560 association parameters. Independent
`target_44b6` checkpoints improved external ZSNS005 selection AP from
`0.173924` to `0.277423` and `0.219762`. Their frozen ensemble selected four
of five true events at `0.80` precision on selection. On the disjoint audit,
however, the same threshold selected seven events with only one true event
(`0.142857` precision). Event classification itself failed; the result was not
a second-daughter ranking artifact. The focused external gate is rejected and
no model dataset, candidate, or submission was created.

## Real-domain division probe and next lane

The clean 0.940 prediction graphs contain 225 constrained geometric recovery
candidates in the five annotated division frames. Exactly three are
parent-free recoveries that require only one added edge. A train-only probe
cache now contains the 15 context frames (`t-1,t,t+1`) needed to score all 225
candidates; its 23 Kaggle members total 66,682,915 bytes and every member is
SHA-256 recorded. It reads no competition test file and is diagnostic-only.

The probe also records inference-available branch geometry: existing- and
second-child displacement, daughter opposition, step balance, midpoint drift,
and constant-velocity midpoint error. A fixed biological symmetry score ranks
the three safe positives `2/26`, `1/2`, and `4/84` within their respective
event frames. This is useful complementary evidence, but not precise enough by
itself to authorize recovery.

The next training lane replaces external-only calibration with legitimate
competition-train supervision. A private CPU-only Kaggle packager is staging
all 199 train GEFF annotations; GPU, TPU, internet, inference, and submission
are disabled. The resulting reciprocal inventory will exclude the four final
probe movies, train on real two-daughter parents plus annotated one-child
parents from the same frames, split optimization/selection deterministically
within each source embryo, and run all model optimization on the Antelume A10G.
This supersedes the rejected external-only policy while preserving the
non-replica and no-metric-hack constraints.

The annotation inventory is now complete. Across all 199 official train
movies it finds 151 two-daughter parents; the four final-probe movies contain
five and remain excluded. The reciprocal training pool therefore contains
146 positive division events and 1,247 same-frame annotated one-child PU
controls. A deterministic division-stratified split holds out 5 of 24 usable
`44b6` events and 26 of 122 usable `6bba` events for model selection, leaving
19 and 96 respectively for optimization. The image extractor is a private,
CPU-only Kaggle job; model training and scoring remain assigned exclusively to
the Antelume A10G.

The private extractor completed as Kaggle kernel version 3 in 57.35 seconds
on CPU (`gpu_used=false`). It produced 146 hash-verified NPZ shards with 1,393
rows: 115 positives plus 1,039 PU controls for optimization and 31 positives
plus 208 controls for movie-disjoint selection. The 34,245,696-byte archive
SHA-256 is `46db151310808fa455e63a4c8791f14121125ba6bda0d46b5aa2b1d7913990e0`;
the manifest SHA-256 is
`943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e`.
Every shard size and digest was rechecked after download. No final-probe image
was included.

## Independent CPU morphology control

A project-authored 132-feature temporal morphology control was trained on the
same movie-disjoint inventory without GPU use. Five-fold grouped optimization
selected a histogram-gradient model and an extra-trees model; their best
individual out-of-fold AP was `0.632866`. The frozen ensemble achieved
selection AP `0.688111` and recovered 8 of 31 positives before the first false
positive, freezing probability threshold `0.8992587384`.

On the complete-movie probe, all three safe recoveries ranked first within
their frames and occupied the top three scores across all 225 candidates
(`AP=1.0`). Absolute probabilities shifted downward, however, so the frozen
threshold made zero decisions. The CPU gate is rejected as a standalone
policy. Its unusually clean ranking is retained only as an independent
diagnostic or conservative vote for the Antelume model; the held-out result is
not used to lower its threshold. Model SHA-256 is
`dc21d9ba50e5fbeb339ecea5ab2842afab4bd7c82c35d40c227ce95ff3be8dc6`.

## Real-domain Antelume gate outcome

The first real-domain run trained only the classification heads for 2,000
steps per fold on the Antelume A10G. It did not produce a two-fold blend that
met the frozen zero-false-positive selection gate and was rejected without
opening the final probe. A subsequent focused run unfroze 25,178,047
division-relevant parameters, kept 21,208,560 association parameters frozen,
and trained each independently initialized fold for 3,000 steps on the same
movie-disjoint split.

The focused run improved selection AP to `0.565943`. Its precommitted blend
selected only the `target_6bba` fold (`target_44b6=0.0`, `target_6bba=1.0`) and
froze logit threshold `1.4140625`; this recovered 9 of 31 selection positives
with zero false positives. Both checkpoints and their blend weights were
SHA-256 bound before opening the final probe. The scorer was repaired before
that opening so it used the frozen weighted blend rather than an accidental
two-fold mean.

The one-time probe ran on an NVIDIA A10G. The frozen model ranked the three
safe complete-movie recoveries `1/26`, `1/2`, and `3/84` within their event
frames and achieved `AP=0.632479`. Absolute logits shifted below the frozen
threshold (`1.091797`, `1.202148`, and `0.439453`), so both threshold-only and
the biological-geometry conjunction selected zero rows. The policy therefore
failed its precommitted graph-evaluation gate and is rejected: no model dataset,
candidate attachment, or submission is authorized from this run.

The consistent pattern across the independent CPU morphology control and the
Antelume deep gate is useful but narrow: division ranking transfers, absolute
calibration does not. The next lane must precommit a movie/frame-scale-robust
decision rule using selection data only and validate it with grouped held-out
evidence. The opened complete-movie probe will not be presented as untouched
evidence for that new policy.

## Hard-negative expansion outcome

The next CPU extractor expanded the real-domain inventory to 2,768 rows in
306 hash-verified shards: 146 division positives, 2,622 controls, and 160
explicit no-division frames. The 67,056,711-byte archive SHA-256 is
`b148a16eef851380c82185b70209e581ac3f7f6c2c2a22362747a0b6d144a605`;
the manifest SHA-256 is
`bfa974f7c5cfa7b2271b7f65128ac9896d458b3e1af150efb238cc5937a4d251`.
The audit role remained sealed while two independently initialized 46.4M
models trained for 4,000 steps each on the Antelume A10G.

The expanded deep run was rejected at selection: target-fold APs were
`0.40855` and `0.47263`, below the pooled and per-embryo gates. An
embryo-specific calibration did not rescue the `6bba` fold. The audit was
therefore never opened for either deep policy. A separate morphology-v2 run
passed selection (`AP=0.699484`, 5 TP before the first FP), then failed its
single audit opening (`AP=0.404859`) and selected zero events after the same
absolute-scale shift. These outcomes strengthen the conclusion that adding
hard negatives alone does not solve probability calibration across movies.

## Scale-invariant ranked consensus

A threshold-free rule was evaluated as a development experiment on the four
complete-movie public-control graphs. Candidates first pass the fixed
biological geometry floor of `3.0`. Within each movie, the eligible parent
ranked first by the independently trained Antelume deep gate must be the same
parent ranked first by the independently trained CPU morphology ensemble. At
most one parent-free edge may be added. No probability or logit threshold,
leaderboard result, node change, coordinate change, or edge reassignment is
used.

The two voters are individually useful on the complete-movie probe: deep AP is
`0.632479` with safe-event frame ranks `1/26`, `1/2`, and `3/84`; morphology
AP is `1.0`, with all three safe positives ranked above every control. Their
rank agreement recovered all three safe missing divisions with `3 TP / 0 FP`.
Adjusted edge Jaccard improved from `0.919680404` to `0.920941968`; division
Jaccard improved from `0.0` to `0.6`. The development artifact is
`.biohub/results/competition-ranked-consensus-division-development-v1.json`.

This result authorizes one full candidate evaluation, not direct submission.
The private runtime dataset binds both model hashes, every project source file,
the development evidence, and the exact scikit-learn 1.9 CPython 3.12 wheel.
Dataset manifest SHA-256 is
`c837f9ba77303dc3393b9d2dd1e531b9f147718721d9bf011e977448fbb80888`.
The attributed EMA candidate is Kaggle kernel
`indarkarhana/biohub-ema-ranked-consensus-candidate-v1`, version 1. It uses two
T4 GPUs, has internet disabled, and contains no competition submission call.
Promotion still requires successful completion, a distinct output hash, no
graph mutations beyond the agreed edges, preserved edge quality, improved
division quality, and a clean runtime receipt.

## Large pretrained voter admission

An independently pretrained 87,640,009-parameter Swin3D-B was admitted to a
standalone ranker experiment on Antelume. The first run exposed a strict
parameter-routing defect: torchvision's final transformer stage is
`features.6`, while the precommitted prefix named `features.7`. Only the
1,025-parameter replacement head trained. Its nested tuning AP was `0.206288`
and its one-shot audit AP was `0.153803` (`44b6=0.768333`, `6bba=0.125109`).
The checkpoint SHA-256 is
`2690f0c68db2a3013503edbbb8acbdf503ca5d3b3a48bcfec8f306e5785098c1`.
It is rejected permanently and cannot vote in any ensemble.

The corrected v2 experiment has a hard parameter contract: 87,640,009 total
parameters and exactly 25,357,761 trainable parameters from the final Swin
stage, terminal normalization, and head. To avoid reusing the audit opened by
v1, v2 reads only the upstream optimization role and freezes a new disjoint
train/tuning/audit partition by complete movie. The upstream selection movies
are not read. The nested audit remains acceptance-only, the decision rule is
ranking-only, and a pooled AP below `0.55`, either embryo AP below `0.40`, or
tuning AP below `0.60` rejects the voter.

The corrected v2 run trained all 25,357,761 intended parameters for 1,500
steps on the Antelume A10G. Tuning AP improved from `0.116867` to `0.368906`;
the fresh nested audit reached `0.490096` (`44b6=0.95`, `6bba=0.469257`).
This is materially better than head-only v1 but below both the `0.60` tuning
and `0.55` pooled-audit gates. Checkpoint SHA-256 is
`06ccb6d00e5e66276092304635e752a4388b3a6ee432c1ddc7ebeff589cf2369`.
V2 is rejected and is not attached to the current candidate. The result also
confirms that network size alone is insufficient: the smaller project deep
gate remains the stronger transferable ranker on the relevant complete-movie
probe.

Candidate kernel version 1 failed after the dual-T4 base inference completed
in 9.33 minutes. The error was isolated to the inserted setup cell:
`INPUT_ROOT` was referenced before the base notebook defined it. No ranked
post-processing, evidence file, CSV promotion, or competition submission ran.
Version 2 replaced that implicit dependency with an explicit immutable
`Path("/kaggle/input")` root, then exposed an incomplete private runtime
package after the same 9.35-minute base inference. The multiscale scorer
imported `contextual_pair_fusion`, which was not included, and stopped in its
setup cell before either voter loaded. Again, no graph edit, evidence file,
CSV promotion, or competition submission ran.

The runtime package now includes the complete transitive source closure:
`contextual_pair_fusion.py`, `pair_fusion.py`, and `transition_context.py` in
addition to the already packaged project modules. A strict isolated import
loaded the exact 46,386,607-parameter deep model and both morphology
estimators. The final package was then rebuilt from a pristine directory and
published with directory uploads disabled, so no generated `__pycache__`
artifact is attached. Its 14-file manifest SHA-256 is
`c837f9ba77303dc3393b9d2dd1e531b9f147718721d9bf011e977448fbb80888`.
The model, policy, and decision rule remain unchanged for the next candidate
run.

Kernel version 3 was pushed after the pristine private dataset reported
`ready`, its published inventory exposed exactly the 14 manifest files, all 25
notebook code cells compiled, and the focused promotion suite passed 13 tests.
It retains the Biohub competition source, the four required private inputs,
GPU execution, and disabled internet. Version 3 completed successfully in
2,097.53 seconds. On the hidden test graphs it scored 41 geometry-eligible
parents, found one deep/morphology rank agreement, and added exactly one edge
without a reassignment, node change, or coordinate change. Submission SHA-256
is `5d57a87a4c3869afd874bc49a289a74a46329f5cba4dab21b1c0ffc5b25250f6`.

The external promotion gate rejected that output. On the four held-out
validation movies, the rank rule added one edge to `44b6_12dfb391` but recovered
no true division. Adjusted edge Jaccard improved by `0.000516`, while removing
the clean control's safe-division behavior reduced division Jaccard to zero and
made total proxy gain `-0.013770`. The full policy therefore fails both the
required positive division gain and the `0.005` proxy-gain floor. No promotion
report or submission receipt was written and no competition submission ran.

The first controller download attempted to mirror the notebook's temporary
scikit-learn installation and hit a Windows path error after the important
artifacts had downloaded. The five required evidence files were then fetched
directly with a version-pinned filename pattern, and the verifier independently
reproduced the same model-policy rejection. The controller is repaired in
commit `e633024` to request only those five files in future runs. Version 3 is
retired; enabling the clean base safe-divider alone would restore the control
but the observed rank addition supplies only `+0.000516`, still below the
promotion floor, so an unchanged rerun is not justified.

## Overnight strong-member ensemble program

The refreshed AWS inventory contains one available GPU host: the Antelume
`g5.xlarge` with one NVIDIA A10G (23 GB). No other running EC2 instance in the
account has a GPU instance type. A hash-checked overnight run is therefore
scheduled sequentially on that A10G rather than falsely claiming multi-host
parallelism.

The run trains eight new seeds from each of the two independent real-domain
initializations: 16 networks total, each with 46,386,607 parameters and
25,178,047 trainable parameters. Every network receives 50,000 optimizer steps
with EMA and best-checkpoint retention, so the longer budget cannot overwrite
an earlier stronger checkpoint. The first worker started successfully with
the A10G at 96-97% utilization. The frozen data-manifest and initialization
hashes were reverified before training.

Admission is deliberately stricter than simply averaging every run. Each
network is scored independently on the movie-disjoint selection split and is
rejected unless pooled AP is at least `0.55`, AP is at least `0.40` for each
embryo, and it recovers at least two positives before the first false positive.
Unique effective seeds and exact training configurations are verified for all
workers, and byte-identical output checkpoints are deduplicated before scoring
so repeated initializations cannot masquerade as ensemble diversity.
Only the surviving models enter one precommitted equal-weight average of their
within-movie percentile ranks. No ensemble-weight search is allowed. That
ensemble is eligible for a later development probe only if it improves AP by
at least `0.01` over the strongest admitted individual and keeps each-embryo AP
at or above `0.45`. Independently, a single admitted network may progress even
if the ensemble does not, but only if its AP exceeds the frozen current voter's
`0.565942529` by at least `0.01`; this prevents an averaging failure from
hiding a genuinely stronger component. The overnight evaluator reads no final
probe, competition test input, leaderboard result, public prediction, or
submission path.

Launch and admission code are committed as `4cbc0f0`, `aa74973`, and
`5ddff9c`. The remote training process and its dependent admission controller
are both active; the observed throughput projects completion near 06:00-07:00
America/Chicago.

A second dependent controller is prepared for the post-selection boundary.
It opens the train-only development probe only when the selection terminal
authorizes either the full admitted equal-rank ensemble or the single strongest
predeclared individual. It cannot search weights, subsets, or thresholds on
that probe, and a selection rejection writes a skip terminal without reading
any probe frame. The downstream graph evaluator now accepts this hash-bound
probe format and starts from the complete base graph, preserving the clean
control's existing safe divisions while evaluating only additive recoveries.
Fourteen focused tests pass across selection, deduplication, calibration-free
ranking, probe routing, and graph-policy eligibility. The scorer and remote
controller are committed in `0ee9d7c` and `20d0ff6`.

The most recent AWS control-plane session expired before that second controller
could be copied to the instance; this does not affect the already running
training or admission processes. A hidden local deploy-on-refresh controller,
committed as `0287326` and repaired for expected expired-token stderr in
`d95ecaa`, now checks for a valid session at 60-second intervals
and will copy and launch the controller exactly once when credentials become
valid. It is restricted to `/home/ubuntu/biohub*` and includes no Kaggle
submission command.

The post-selection implementation is now end-to-end rather than probe-only.
Commit `557e2a2` centralizes the immutable member policy and adds a runtime
packager that hash-binds the selection terminal, member-policy source, one
probe opening, graph-development evidence, morphology voter, and every unique
deep checkpoint. Its policy requires the clean base safe-divider to remain
enabled and permits the learned stage to make additive edge edits only. Commit
`2fc8b2f` adds the offline two-T4 Kaggle notebook builder and external
promotion verifier for that runtime. The notebook averages calibration-free
within-movie ranks across all admitted deep members, then requires agreement
with morphology; it does not average weak or duplicate models. Commit
`16c9cd1` adds the exactly-once submitter and durable candidate controller.
That controller can submit only after the actual full notebook improves the
frozen validator proxy by at least `0.005`, increases true divisions and
division Jaccard, respects the edge-regression limit, preserves the base rule,
and differs from every audited public submission hash. Commit `7119a4b`
extends the credential-refresh deploy chain to ship and hash-bind the shared
member-policy source. None of these downstream stages will launch if the
overnight selection evidence rejects all models.

Commit `9aea19c` adds a second local event controller for evidence recovery.
After deployment, it authenticates once and holds one SSH session while the
remote controller waits server-side. The remote side streams a gzip archive
containing the selection terminal, probe/controller evidence, and only the
checkpoints named by the precommitted probe policy. Every archived file has a
size and SHA-256 record in an internal manifest; the local verifier checks all
records without extracting the archive and rejects duplicate or path-escaping
members. The deploy and harvest watchers are both live. This removes repeated
remote status polling and preserves the exact accepted artifacts even if the
short-lived AWS session expires again after the SSH connection opens.

The first durable worker checkpoint, seed `205043` / `target_44b6`, completed
all 50,000 steps in 1,708.51 seconds and retained step 12,000. It improved its
initial AP from `0.243307` to `0.468185` and recovered 4 of 31 positives before
the first false positive, but it remains below the `0.55` independent-member
floor and is therefore rejected from the ensemble. Checkpoint SHA-256 is
`3c8dd8c2dc7d989abbdc7b995a52c1c00bed86afd7f04b8d74b3e965474c4371`.
The paired `target_6bba` worker continues; no probe or submission evidence has
been opened by this sweep.

The current primary-method refresh supports this conservative admission
policy. The official Trackastra project remains the strongest directly
relevant general-purpose learned linker and now advertises an optional
SAM2-feature variant, but that released feature path is described for 2D data;
it is not silently treated as a validated 3D zebrafish model here. The 2025
ICCV study *How To Make Your Cell Tracker Say “I dunno!”* reports that learned
trackers, including transformer trackers, can be overconfident under temporal
and domain ambiguity and that calibrated uncertainty is useful. Its cheapest
calibration method needs representative ground truth. That evidence argues
against transferring an absolute validation threshold to hidden embryos and
supports the current within-movie rank plus independent-consensus policy.

Primary sources:

- [Trackastra official repository](https://github.com/weigertlab/trackastra)
- [Trackastra paper](https://arxiv.org/abs/2405.15700)
- [ICCV 2025 uncertainty-aware cell tracking paper](https://openaccess.thecvf.com/content/ICCV2025/html/Paul_How_To_Make_Your_Cell_Tracker_Say_I_dunno_ICCV_2025_paper.html)

## Late frontier and account refresh

A fresh score-descending Kaggle kernel inventory on 2026-08-29 remains headed
by notebooks explicitly titled as metric hacks; these are excluded. The most
recent clean-looking public notebooks inspected claim `0.928`, `0.935`, and
`0.937`. Their code remains in the same dual-seed harmonic-fusion family and
uses hand-set safe-division, gap, and short-track heuristics. None supplies a
clean, independently gated component above the already reproduced `0.940` EMA
control, so none is copied into the project candidate.

The account submission inventory was also refreshed. The last visible scored
project submissions remain in the `0.912`-`0.913` range; the attributed clean
control submitted on 2026-08-26 is complete but currently has no displayed
score. There were zero UTC-day submissions before the ranked-consensus
controller started. The controller therefore has quota headroom but still
requires all local promotion gates before it can submit.

## Exact development-baseline binding

The archived ranked-consensus development result was independently replayed
against each complete cached clean-frontier graph set. The `0.937` base and
`0.938` harmonic states were rejected because the proposed recovery edge was
already present. The path previously labeled as processed public control was
also rejected: it already contains `44b6_267148e4/646->840`, so it is a later
graph state and cannot serve as the pre-addition baseline. Only the exact
`0.940` EMA graph set reproduced the archived result: three selected edges,
three true recoveries, zero false recoveries, pooled edge Jaccard
`0.9196804037 -> 0.9209419680`, and division Jaccard `0 -> 0.6`.

The new baseline verifier binds the tree SHA-256 for all four EMA predictions
and all four truth graphs, both original probe hashes, the evaluator source,
the archived evidence, and the byte-exact replay. It also requires substantive
equality between the archived and replayed JSON while allowing only the two
new probe-hash fields added by the current evaluator. The verified descriptor
is submission-ineligible and records the rejected ambiguous cache explicitly.
Three focused tests cover the complete contract, graph drift, and substantive
evidence drift.

The verified harvest now feeds a third durable event controller. It first
re-verifies the complete streamed archive and manually extracts only regular,
hash-matched members into a new archive-hash-named directory; neither tar path
resolution nor unverified bytes can write to disk. Archive and checkpoint
hashes are streamed in fixed-size blocks so a 16-member harvest does not need
multi-gigabyte host memory. A rejected overnight selection stops at a local
skip terminal. A completed selection must pass the exact EMA baseline
contract, then receives one additive ranked-consensus development evaluation.
Only a positive result can invoke the strong-member runtime packager with the
pinned CPython 3.12 scikit-learn wheel. This stage includes no dataset upload,
Kaggle kernel launch, or competition submission command. Eight focused tests
pass across harvest verification/extraction and strong-member packaging.

The final launch boundary is also event-driven. If and only if the local
post-harvest terminal says `runtime_packaged`, the launch controller verifies
the runtime-manifest hash, creates or versions the private v2 dataset, waits
for Kaggle processing, builds the attributed additive candidate, and requires
private/offline GPU metadata plus the notebook's exact two-T4 runtime guard.
It resolves the next owned kernel version before pushing, validates that exact
version and its attached competition/runtime sources after launch, then starts
the existing external verifier/submitter with the same runtime manifest and
kernel version. The launch layer itself cannot submit. A scientific rejection
therefore creates neither a dataset nor a kernel, while a technically complete
but weak notebook is still stopped by the proxy and true-division promotion
gates before the exactly-once submitter can run.

## Candidate-aligned relational division lane

The v2 hard-negative experiment was not repeated: its two 46.4M deep models
had already been rejected at selection. Instead, a new local simulator uses
all 199 hash-verified official train GEFFs to formulate the exact additive
decision made at inference. A positive example retains one true daughter and
proposes the other; hard negatives pair the same parent/retained-daughter
context with nearby non-child proposals. Parent, retained-child, and proposed-
child centers share the same `t-1,t,t+1` temporal window, and the downstream
model contract is invariant to daughter order. The deployed geometry rule is
still frozen at `>=3`; a broader distance-valid curriculum is retained for
representation learning and each row records whether it is inference-
eligible.

The deterministic movie-disjoint inventory contains 3,013 relational pairs
from 146 non-probe movies: 134 true division pairs and 2,879 candidate-specific
hard negatives. Of these, 55 positives and 165 negatives satisfy the final
geometry gate. Inference-eligible negatives are priority-retained before
broader curriculum negatives, preventing the rare 44b6 cases from being
silently removed by the per-movie cap. Optimization, selection, and sealed
audit roles each contain both classes for both embryos; the four complete-
movie final probes remain excluded. Inventory SHA-256 is
`94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e`.
Five focused tests cover positive/distractor construction, distance rejection,
disjoint role allocation, negative-only role supplementation, and eligible-
negative retention.

The inventory is now published as private dataset
`indarkarhana/biohub-relational-division-inventory-v3`, version 1. Its internal
manifest SHA-256 is
`c5f093601d74201739e003f95eb56511b33df472d4e5683f7354584bd086ee23`,
and Kaggle reports the dataset `ready`. The first Windows upload attempt failed
before creation because the CLI encoded a relative directory into its temporary
upload filename; running the same create operation from inside the staging
directory avoided that client bug without changing any bytes.

A CPU-only, offline v3 extractor consumes that exact private manifest and the
official competition train source. For each relational row it samples the same
three frames at the parent, retained-daughter, and proposed-daughter centers,
producing `(N, 3 centers, 3 temporal channels, 17, 17, 17)` patches plus nine
physical geometry features. Inference-eligible negatives receive full label
weight; broader curriculum negatives remain downweighted. Audit features may
be materialized, but the extractor never scores audit labels, opens the final
four movies, attaches test data, or includes a submission command. Five
focused inventory-package and extractor tests pass.
