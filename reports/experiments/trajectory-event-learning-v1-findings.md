# Event-structured learner: evidence and next checks

## 2026-09-14 real-data functionality and full-movie smoke

The fixed three-epoch source smoke completed on four source frames from
`6bba_6ca87370` and `6bba_2819ca14`. Its 42 known parent constraints remained
feasible under the joint event solver. Mean source hinge fell from 0.38958 to
0.24596; known-link mistakes on those four frames stayed at three. No test or
selection labels were opened. Twenty-four focused core, training, supervision,
candidate and full-graph inference tests pass.

An additional complete 100-frame diagnostic used the same two training movies.
The patched micro-score increased from 0.95769218 to 0.98838439. Edge TP rose
1945 -> 1978, FP fell 52 -> 22, FN fell 56 -> 23. Division counts were unchanged:
0 TP / 0 FP / 2 FN. All cells and coordinates are exactly unchanged. This is
training-movie evidence, **not** an independent quality gain or leaderboard score.
The inference stage completed all 99 transitions per movie without solver
fallback or budget exhaustion, in 17.88 and 8.52 seconds on the local CPU.
It changed ordinary association decisions; it has not solved the division gap.

Freeze checkpoint SHA-256
`508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be`.
Do not refit or select another epoch in response to the following diagnostic.
Next test all eight batch-0 source movies, reporting the six event-head-unseen
movies separately, each embryo, every movie, runtime fallback, and division
errors. One of these six is source44, so embryo evidence is necessarily narrow.
Public pretrained backbones and prior pipeline stages are not claimed to be
independent of all competition training data. These remain source diagnostics;
selection and eight-movie validation are still required before promotion.

Source collections batch 0 and 1 are complete and exactly backed up. Their
physical supervision totals 14,882 available parent constraints and 30 annotated
divisions. The planned optimization corpus remains all 68 source movies; no
hard movie is removed based on model performance. Batch 2 is running on
Antelume under the existing sequential controller, with batches 3-7 queued.
Case preparation visits every source-supervised transition and persists bounded
per-frame arrays so the full fit need not hold the entire corpus in RAM.

Batch 0 case preparation completed in 264.84 seconds: 772 feasible frame cases,
6,251 retained known-parent constraints, 141,141,016 bytes of compressed inputs,
and no incompatible partial-label frames. Ten known constraints were explicitly
outside the editable graph scope; additional labels on protected children never
enter these problems. Batch 1 case preparation is now running locally, separate
from the Antelume GPU collection. The frozen eight-movie diagnostic is also
running locally. At 14:29 UTC, Antelume PID 61886 was live and GPU utilization
was 100%; no extra GPU experiment was launched.

Submission 56231458 remains untouched. No event checkpoint is submitted or
authorized for submission by these diagnostics.

## Frozen smoke head: broader source diagnostic

Eight-movie score: 0.93397223 -> 0.95334422. The six movies unseen by the event
head improve from 0.91561264 -> 0.93024063. Both embryo aggregates are
nonregressing; source44 has only one movie and is neutral. Six movies improve,
one is neutral, and `6bba_5c039895` regresses by one extra false link (no TP
gain). Therefore the strict movie gate **fails**. No manual rule, per-movie
router, score-based epoch choice, or promotion is authorized by this result.
Division counts remain 5 TP / 8 FP / 7 FN across all eight movies.

The dense movie `6bba_3abfe10a` processes only 58 of 99 transitions before the
two-minute stage cap; one solver timeout also preserves its incumbent. All
100 frames remain in the valid scored output, with later links unmodified.
Do not claim complete refinement coverage or transfer this local wall time to T4.

Vectorized geometry passes exact float32 replay for all 772 frozen source cases:
102,432,780 feature values, no discrepancy. The eight-movie audit takes 49.92 s
including candidate generation, compressed case reads, integrity checks and
comparison; feature calculations themselves total about 3.13 s. It changes no
weights or features. Full-graph and deployment equivalence are still unverified.

Next bounded fit, frozen before outcomes: use all 673 prepared frames from the
seven source6 batch-0 movies, three epochs with the original optimizer settings,
and initialize from structured v1 plus source-only annotated division odds—not
from the smoke checkpoint. All source44 and all eight batch-1 movies remain out
of this new event fit. Twenty-minute CPU deadline and per-epoch checkpoints;
Antelume continues its single source collection job. This is an early broader
data pilot, not a replacement for the planned 68-movie corpus. Validation uses
the final fixed epoch; no retry or threshold search is authorized by this plan.

## Runtime and broader data progress

The seven-movie fit completed epoch 1 (673 optimizer steps) at 796.41 seconds;
checkpoint SHA `1a822b9884a85ff1638f99618fac9919ac6f16d43072c5678eeb2908074c1ac2`.
Do not promote that partial epoch or call training complete. The three-epoch
contract and twenty-minute cap remain in force; inspect the actual job for its
terminal state. Source-only case preparation also completed batch 1 (787 cases,
8,478 retained constraints) and batch 2 (774 cases, 6,775 retained constraints).
Across the first three new batches: 2,333 cases / 21,504 retained constraints.

Accelerated full-movie replay is exactly equal on both small complete movies
and on all 57 comparable nonfallback transitions of the dense movie. However,
the dense movie still hits 120 seconds: 77 transitions processed, nine solver
fallbacks. Full refinement coverage is therefore **not** established. Geometry
vectorization alone is not sufficient, and no release claim follows from it.

An exact dominance reduction now removes a fork only when an allowed
continuation plus an allowed birth consume the same cells with strictly greater
total score. Forbidden births cannot prune annotated two-daughter constraints;
ties are kept. Four focused tests pass, including 25 random competing-fork
problems with partial masks and equal optimal objectives.

On the eight largest frozen source problems, the original two-second solver
finishes optimally on four, whereas the dominance-reduced solver finishes on
all eight. All four comparable objective values agree exactly. Reduced solve
times are 0.83-1.30 seconds plus 0.09-0.19 seconds of reduction. This is a bounded
source runtime benchmark, not yet full-movie/Kaggle acceptance. No weights or
probability thresholds were changed. Next: integrate the proven reduction into
a separately versioned inference runtime and test full-movie coverage, preserving
all source score failures and avoiding model/epoch selection on held-out labels.

Antelume batch 3 is active under the original collection controller. Its live
PID is 65151 and remote root is `/dev/shm/biohub-event-source-v1-b3.hpf9EX`.
Earlier roots/results remain backed up; do not relaunch an observed running job.

## Completed broader fit and source-selected anchor diagnostic (September 14)

The first seven-movie pilot reached its twenty-minute deadline after one epoch;
its partial checkpoint is not a candidate. The separately versioned fast retry
completed all three predetermined epochs in 662.265 seconds, using the same
673 cases, initialization, optimizer, order, seed and regularization. Final
weights SHA `01c7ad5c6078db669b8e7042934a23cd3bc2534c00f850681f89adde48afada1`;
contract SHA `c738ea876106a40cf9b0f57877e12644dc17d3ff6b9dc6c184a717ea975279c1`.
Epoch one exactly matches the slow run's checkpoint. Constructive feasibility,
bounded case caching, exact fork dominance and one BLAS thread account for the
engineering changes; no epoch was chosen based on quality. Its complete-movie
audit covers seven fitting and nine event-head-held-out source movies and is
running under session 68977. No selection/validation data entered that fit.

The anchor ablation completed: frozen old 18 edge weights plus the source-only
division prior, without fitting the new event coefficients. The artifact is the
smoke NPZ SHA `508165d131dc0c2516cf91e0eafb5a88edf24358f6ced5b0436f29ccc6b733be`,
but **parameter key `anchor`, not `weights`**. Eight source movies improve from
0.93397223 to 0.95601924; six improve, two are neutral, no movie regresses.
The six not used for the smoke's source prior improve to 0.93419864 from
0.91561264. Division counts remain 5 TP / 8 FP / 7 FN; cells and coordinates
are unchanged. All 99 transitions on each of these eight movies completed
without solver fallback. This suggests the expanded assignment freedom drives
much of the gain; it is not proof of leaderboard transfer or division recovery.
Source audit SHA `e1667c04fd1dd83d18932f63981e98e54ff0a2f72873932f55216987131b44fd`.

Freeze this one source-selected anchor variant for the ten role-disjoint
selection movies. Inputs were generated before opening new selection labels;
receipt SHA `75258e792ca5eeb2d9763c20a43d2478ee55aa31ddbef0ef2ea596aa4f660d11`.
The new audit freezes every graph before scoring and checks complete finite
scoring, pooled/raw-edge improvement, every movie, both embryos and division
counts. Five focused quality-check tests pass. Prior pipeline exposure to this
selection set is disclosed; public-backbone independence is not established.
No per-ID rule, learned gate, threshold search, or selection refit is involved.
Passing selection alone does not authorize submission: separate validation and
offline runtime/portable acceptance remain required.

The tiny trained head with exact pruning scores 0.95376901 on the source eight,
but still adds the one false link on `6bba_5c039895`; that failure remains visible.
Local dense-movie throughput varies with CPU contention: an isolated replay
finished all transitions in 80.27 seconds, while another audit reached only 94
before its 120-second cap. Full output movies remain intact via baseline fallback.
Also, fewer than 99 generated problems can mean no editable scope at some times,
not necessarily an incomplete movie; distinguish this from budget exhaustion.

The original source controller stopped after an SSH observation failure, not a
remote GPU failure. Recovery observed the same batch-3 PID, harvested it without
restarting, and has now completed batch 4. Session 94626 owns batches 5-7.
At the direct September 14 check, batch-5 PID 71072 was live on Antelume and GPU
utilization was 100%. Completed batch-4 images (3,220,793,845 bytes) were retired
after output verification and remain recoverable from the competition archive;
experiment outputs and foreign jobs were preserved. Batch-3 local supervision
preparation is running separately. No other GPU experiment was launched.

Authenticated Kaggle check still reports submission 56231458 PENDING, with no
score. Previous completed submission 56219125 is 0.946. These experimental source
scores must not be reported as new public scores.

### Broader pilot terminal result and independent anchor test launched

The 16-movie pilot audit completed in 574.563 seconds. All 16 movies are included
in adjusted scoring. Pooled score 0.89888186 -> 0.92054704; nine movies unseen by
this event fit improve 0.87902414 -> 0.89934779. However, `44b6_e28840c6` drops
0.00365832 and `6bba_5c824876` drops 0.00165423. Source44 aggregate also drops
0.99138787 -> 0.98971121, so the movie and embryo gates fail. Division totals
remain 6 TP / 11 FP / 24 FN. No coefficient selection, source-ID exception,
promotion, or submission follows from this result. Two movies exhausted the
refinement budget (95 and 57 processed transitions); their remaining baseline
edges and all 100 frames were preserved. Another movie generated 97 editable
problems without exhausting its budget, which is not a truncation failure.

The fixed anchor selection audit is now running under session 60837 using
`scripts/audit-trajectory-event-anchor-selection-v1.py`, after the broader pilot
terminated. This avoids concurrent local model-audit contention. It uses the
already frozen `anchor` parameter key, not the broader fitted weights; no
threshold or model choice will be tuned against these ten labels. The five
quality-check tests and a real source-report serialization check pass.
Batch-3 physical supervision also completed: 7,914 known-parent constraints,
66,170 safely distinguishable wrong-parent options, and 20 annotated divisions.
Its training-case preparation remains outstanding; no duplicate job is queued.

## Fixed anchor selection: gain confirmed, subgroup failures preserved

The ten-movie selection audit is terminal, SHA
`43e94177fe98c2ade5925aa78fd5c892a420a7d8323d1dfa986c75885d240fcb`.
Score improves 0.91193018 -> 0.93646543; raw edge Jaccard improves
0.89033484 -> 0.91486828. Seven movies improve, one is neutral, two regress.
All ten have complete finite adjusted scoring, unchanged nodes/coordinates and
99 processed transitions without solver fallback or deadline exhaustion.
Division counts remain 5 TP / 6 FP / 12 FN.

Failures: `44b6_706092f0` loses one correct link and adds one false link
(105/13/12 TP/FP/FN -> 104/14/13), score -0.01459029 on its sparse annotation;
`6bba_4ffd3da3` loses three correct links while removing one false link
(651/28/38 -> 648/27/41), score -0.00287920. Embryo44 aggregate drops
0.94054558 -> 0.93868408; embryo6 improves 0.90894718 -> 0.93587517.
The strict movie and embryo gates therefore FAIL. Do not call this release-ready
or reinterpret the unchanged division counts as improved division recognition.

Autonomous review permits one **diagnostic-only** evaluation of this exact
unchanged variant on the separate eight T4-cache movies, because the pooled
gain is large enough to warrant further generalization evidence. This does not
waive release gates, hide selection failures, change coefficients or thresholds,
or add a source/movie-ID selector. The eight-movie receipt will explicitly retain
the selection failures and `overall_release_gates_pass=false`, even if its own
checks pass. Predictions must be frozen before scoring. No submission is
authorized by this diagnostic. Public-backbone and prior-pipeline exposure
limitations remain disclosed. No validation-based model search is planned.

Label-free eight-movie input preparation is running under session 51323.
Source batch-3 case preparation was launched once (session 90517), after all
selection predictions were frozen. It remains separate from the controller's
Antelume collection. Any concurrent local preparation must be disclosed with
runtime results, and local times are not a substitute for Kaggle acceptance.

## Separate eight-movie anchor validation and deployment work

The fixed eight-movie diagnostic completed, SHA
`f56568612879bf5d7f604a459b17ca2a7f4f6185920f51af8e0cdccfa3d465b1`.
Score improves **0.94839712 -> 0.96542986**, raw edge Jaccard
0.92926894 -> 0.94631410. Both embryo aggregates improve: source44
0.91737419 -> 0.93079110; source6 0.97495891 -> 0.99451627.
Six movies improve, one is unchanged, one regresses. `44b6_12dfb391`
changes TP/FP/FN 745/35/28 -> 743/36/30, score -0.00369636.
Division counts remain 1 TP / 1 FP / 4 FN. All eight adjusted scores are
finite and included; nodes and coordinates remain identical. The strict
per-movie gate still FAILS, as do the recorded selection subgroup gates.
Do not relabel those gates as passing or call 0.96543 a public score.

No movie exhausts its refinement budget or encounters a solver fallback.
The existing `all_refinement_transitions_completed_without_fallback=false`
field is overly strict for `6bba_f1fde7e0`, which has 97 generated editable
problems, not 99. It did not hit a runtime cap and all 100 output frames exist.
An explicit expected-editable-time check is still needed before using a revised
coverage claim. Do not overwrite the original result merely to change the flag.

Review decision: proceed with **deployment engineering of this unchanged
candidate**, preserving selection and validation failures. Large gains in two
distinct evaluation cohorts warrant exact portable replay and offline runtime
testing; this is not automatic submission authorization or evidence of top five.
No tuning on validation, ID routing, or retrospective exception rule is added.
Small-movie-first portable replay is running under session 8461. Its first full
movie (`6bba_23af9eeb`) is exactly identical in 4.02 seconds with no fallback.
The bundle includes only selected prediction functions/classes and 30 frozen
coefficients, with no annotations or fitting routines. All eight exact replays
must complete before staging Kaggle acceptance. Ten focused tests pass.

The paired selection uncertainty diagnostic uses 5,000 seeded whole-movie
bootstrap draws within the two observed embryos. Conditional on the observed
3:7 embryo composition, the pooled 95% percentile interval for score change is
[0.01203, 0.03810]. Source44 interval [-0.01459, 0.00392] remains uncertain.
These are not probabilities of leaderboard improvement and do not establish
new-embryo generalization or replace failed subgroup checks. No parameter was
selected from this analysis.

Source batch 3 cases completed: 747 cases, 7,824 retained constraints, 207,440,981
compressed bytes, 298.281 seconds. Batch-4 supervision completed with 5,346 known
constraints and 12 annotated divisions; case preparation is running once under
session 60661. Controller 94626 has backed up batch 5 (70,125,005 bytes) and
started batch 6 on Antelume, PID 74383, root
`/dev/shm/biohub-event-source-v1-b6.iuDod1`. The controller still owns batch 7;
do not launch duplicate GPU work or disturb other projects.

### Portable terminal and staged offline notebook

Portable replay completed all eight graphs exactly in 265.203 seconds, with no
solver fallback or budget exhaustion. Code SHA
`c7f98a711adb4d3247b12297cde0b31a246043e906d8ea462a3e5f763f03d954`, model JSON SHA
`3a7db02111037732f3130337a68ffe551b4a80ddf04f9b1690c844e292d88c4b`, proof SHA
`da428b0f52c4255d463e684e04856f192c00541b4587aff437427a0daa353ab1`.
An explicit regenerated-scope check confirms `6bba_f1fde7e0` has exactly 97
editable times; 87 and 88 have no editable problem. Processed times exactly match
that scope, explaining the original conservative 99-count coverage flag.

The offline two-T4/four-worker acceptance notebook is staged, NOT launched:
`.biohub/staging/biohub-event-anchor-acceptance-v1`, kernel
`indarkarhana/biohub-event-anchor-acceptance`. Notebook SHA
`c0cc7810a06f4c9b54f5ff23b0ca6f62f3e13d421769a40cc41f309ef6416084`, contract SHA
`e8f1c79e6309fb9458c64e9e879b2576d0d570d61856cffafcb64cb50b6a6801`, overlays SHA
`d975ea7e7579edc185d463c7dc43bc9def4de6e6f543ea842685dfb097b5f623`.
It preserves the structured intermediate graph and records the new event stage
separately. It explicitly limits BLAS threads around the event stage; the
Kaggle environment must verify the threadpoolctl import in the small runtime
acceptance. No production notebook or existing submission was modified.

Fresh authenticated CLI observation reports 28.12 GPU hours remaining (1.88 used),
while submission 56231458 remains PENDING. The previous launcher rejects all
nonterminal competition submissions and also requires an idle Antelume GPU plus
an old terminal-file path. Do not blindly reuse it: that path may have been
retired, and controller 94626 currently owns active source collection. Preserve
the eight-hour reserve and sequential GPU experiment rule. Hidden-evaluation
quota accounting needs authoritative clarification before altering the pending
reservation guard; a community reply alone is not a verified policy change.
Staging is not a launch or a queued automatic job.

Batch-4 case preparation is terminal: 774 saved cases, 5,320 retained constraints,
184,480,368 bytes, 261.016 seconds. One source44 frame (`44b6_eb2880fc`, t=73,
four retained constraints, 12,464 options) is incompatible with the current
event vocabulary. The receipt correctly has `training_allowed=false`; do not
silently drop that frame or use this batch for a full fit before diagnosis.
This does not modify or invalidate the separate frozen anchor candidate.

## Quota accounting corrected from direct staff evidence

The earlier per-physical-device reservation double-counted T4 usage. Direct
authenticated forum retrieval confirms Kaggle staff comment 1995818 on topic
361104: one hour on a T4x2 notebook consumes one quota hour. The comment-content
SHA is `301a1e0f286674a4e343177493be3f193cb34ec294e31c06bdc7c43550e8af2c`.
Source: https://www.kaggle.com/discussions/product-feedback/361104#1995818 .
The 2026 ARC accelerator page also puts T4x2 and P100 in the same rate tier.
Evidence is saved in `trajectory-event-t4-quota-policy-v1.json`; historic launch
receipts remain unchanged. This does NOT assume hidden scoring is free.

New acceptance guard reserves the full twelve-hour platform cap for the known
pending submission 56231458, plus two hours for the one-hour acceptance test,
then requires eight hours to remain. At 28.12 hours remaining, this leaves 14.12
after both reservations. Unknown pending jobs/statuses, nonfinite quota,
duplicate inventories, and reserve breaches fail closed. Eight focused tests
pass. This removes the blanket pending-status blocker while preserving a
conservative outstanding-run budget. The prior submission is not cancelled,
modified, or resubmitted.

`launch-trajectory-event-anchor-kaggle-v1.py --check-only` passes its artifact
and policy checks and is waiting for controller 94626's final batch 7. It will
require the controller terminal, inspect the actual remote terminal/PID/GPU
processes, verify the previous saved Kaggle kernel is complete, and re-query
submissions/quota immediately before push. Foreign project GPU jobs are not
terminated. The acceptance launch receipt is one-shot; do not repeat a push on
an observation timeout. At this point the launcher has NOT pushed a kernel.

The new event release verifier independently checks cell identity, protected
division/synthetic/gap incidence, lineage degrees, edit-count logs, and unchanged
fallback/unprocessed transitions. Five tests pass after correcting a malformed
toy fixture (missing node IDs and an empty final graph); all eight real frozen
graphs also pass these invariants. `verify-trajectory-event-anchor-kaggle-v1.py`
is ready to verify smoke/full stages, exact eight final/intermediate graphs,
reassembled CSV, and the existing eight-hour projected validation headroom.
It preserves all quality failures and requires separate release review.

## Source frame incompatibility diagnosed

The source44 t=73 case is infeasible because the nearest-eight daughter cutoff
excludes the required fork for observed parent 32620 and daughters 32971/32995.
The same unchanged four constraints become feasible with a uniform sixteen-child
fork vocabulary: 12,464 -> 26,112 options. Neither labels nor predictions were
changed. Receipt: `trajectory-event-incompatible-source-v1.json`.
The wider vocabulary is NOT adopted for the frozen candidate and batch-4's
training guard remains false. A future training version needs a consistently
validated vocabulary rather than dropping this frame or giving training-only
access to a fork unavailable at inference.

Batch-5 physical supervision preparation is now running locally; batch-7 GPU
collection remains under the existing controller. No concurrent second Biohub
GPU experiment was launched.

## September 14 continuation: acceptance launched and source scope extended

The preceding user-question turn explained selective routing but made no model
change (no implementation progress). This continuation revalidated the actual
jobs: controller 94626 exited successfully after all requested batches, and
batch-5 supervision exited successfully. All 60 newly collected source movies
now have backed-up outputs and frozen structured features. Antelume's final
batch terminal and process check confirmed no remaining Biohub GPU process;
no foreign process was stopped. The controller retired the exact final
1,896,969,152-byte image cache, recoverable from the competition archive and
saved manifest; experiment outputs remain backed up.

The first acceptance launch attempt stopped before writing a launch receipt or
pushing a kernel because the CLI returned `KernelWorkerStatus.COMPLETE` rather
than bare `complete`. The exact-kernel status parser now recognizes both forms,
rejecting running/error/wrong-kernel/unknown output. Fifteen quota/status tests
pass, and the combined quota/release/anchor-quality/inference suite passes all
30 focused tests. This was a parser correction, not a terminal-check waiver.

At 16:35:16 UTC, the guarded launcher successfully pushed version 1 of
`indarkarhana/biohub-event-anchor-acceptance`. The staged model, notebook and
contract hashes are unchanged. Fresh quota was 28.12 hours; twelve hours were
reserved for pending submission 56231458 and two for this acceptance, leaving
14.12 after reservations, above the eight-hour reserve. This is an offline
two-T4 runtime acceptance, not a competition submission. The prior submission
remained pending at launch; no replacement or cancellation occurred.

Watcher `watch-trajectory-event-anchor-acceptance-v1.py`, session 56797, owns
observation, terminal output download and the existing verifier. It observes
only this exact version, polls at sixty-second intervals, and cannot launch or
submit. A transient observation failure never restarts the GPU job. A download
or verifier failure is recorded for inspection rather than automatically
retried. Its report is `trajectory-event-anchor-kaggle-v1-watch.json`. The live
Kaggle status was RUNNING. No acceptance result is claimed yet.

`audit-trajectory-event-anchor-expanded-source-v1.py`, session 56995, now
evaluates the unchanged anchor on batches 1 through 7 (52 additional source
movies, excluding already scored batch zero). Its pinned full-movie scorer and
artifact checks are reused; all predictions are frozen before annotation
scoring. Neither event weights nor a movie/correction gate are fitted. Both
prior-fit movies are outside this cohort, but independence from public-backbone
training is not claimed. First complete movie passed all 99 transitions without
fallback. Local CPU evaluation does not consume additional GPU quota.

Batch-5 case preparation completed: 767 cases, 4,625 retained constraints,
275,305,270 compressed bytes in 338.891 seconds. One incompatible partial frame
keeps its training guard false. Batch-6 and batch-7 supervision also completed,
with 5,615/3,101 known constraints and 11/6 annotated divisions respectively.
Their old-vocabulary case preparation has not been started.

`trajectory-event-fork-coverage-v2.json` independently checks both known source
incompatibilities: source44 batch4 t73 and source6 batch5 t17. Both are infeasible
under the uniform eight-nearest-child fork vocabulary and feasible at sixteen,
using unchanged labels and prediction-only candidate construction. The second
case grows from 12,131 to 24,913 options. This is a small functionality test,
not complete-source coverage or a runtime proof. No training-only oracle fork,
label deletion, movie-ID exception, or candidate modification was introduced.
The current head and its recorded selection/validation failures remain frozen.

## Selective-correction groundwork, no gate fitted

The previous continuation made progress (actual acceptance launch, completed
source collection, source-vocabulary diagnosis). Its existing jobs were
revalidated by their same handles; no job was restarted. Kaggle version 1
became COMPLETE by the watcher observation around 16:56 UTC. Watcher 56797 is
downloading terminal outputs and will run the verifier; complete runtime and
graph-identity acceptance are not yet claimed.

Added `research/trajectory_correction_gate_v1.py`: changes are partitioned into
connected bipartite components within each transition. Complete components can
be accepted or rejected independently while preserving at-most-one incoming
and at-most-two outgoing edges. Existing division/synthetic/gap incidence and
cell positions stay protected. IDs and transition times locate edits but do not
enter the 38-dimensional feature vector, which contains log edit counts and
means of the existing 18 prediction-only edge features for each side.

The real first-source-batch audit (`trajectory-correction-components-v1.json`)
completed all eight movies in 19.203 seconds. All/none masks exactly replay the
candidate/baseline graphs, and three random masks per movie pass complete-movie
graph invariants. The module includes no trained gate, no chosen threshold,
and no deployment change. Unit tests also cover atomic swaps, fork additions,
protected incidence, unknown feature evidence, ID/time renaming invariance, and
100 randomized graph pairs with every component mask. Eleven gate/supervision
tests pass; this does not establish improved scores.

Added separate source-only correction supervision. It compares known correct
parent links before and after an atomic correction, ignores unannotated child
targets, and excludes ambiguous alternatives within the physical matching
tolerance. An absent parent is an error only when a real annotated parent is
known. These labels describe partial known-link changes, not complete-component
official metric gains, and are never inference features.

The first source-label audit (`trajectory-correction-supervision-v1.json`)
found 2,272 correction components: 60 with net known-link improvement, eight
with net known-link regression, and 2,204 unlabeled or neutral. Sparse labels
cover 83 children, with 3,085 unknown and eight ambiguous child comparisons.
This small, biased observed subset is inadequate to claim a calibrated gate;
no fit or threshold selection was performed.

`prepare-trajectory-correction-source-v1.py` is prepared and syntax-checked to
freeze all 60 source movies' correction features and separate partial labels.
It requires the existing 52-movie source evaluation to finish first, verifies
scope/model/feature/prediction hashes and exact all-components replay, and
never reads selection or validation annotations. It has not yet been run.

A descriptive combination of the existing selection10 and validation8 rows
using the pinned official summarizer gives 0.92568996 -> 0.94759546 across 18
movies. Source44 combined seven movies give 0.93116858 -> 0.94044536; source6
combined eleven give 0.92477717 -> 0.94982021. This post-hoc descriptive pool is
not a new independent cohort and does not erase the original selection embryo
or individual-movie failures. It is not a leaderboard prediction.

An additional web check found community claims that hidden submissions do not
consume notebook quota, but no new direct staff policy evidence was established.
The conservative pending-submission reservation therefore remains unchanged.

Same-handle observation subsequently confirmed all 52 expanded-source
predictions frozen, followed by official source scoring. Session 56995 is
still running that scoring, not retraining. Watcher 56797 remains live in its
single terminal-output download. No source-score total, acceptance pass, gate
promotion, production launch or new competition submission is claimed at this
handoff. The prepared 60-source correction extractor must wait for the source
audit's terminal report before it is invoked.

## September 14 terminal results and public production test

The 52 additional source movies completed official scoring in 1,622.328 seconds:
0.91547762 -> 0.92788191. Both embryo aggregates improve: source44 (12 movies)
0.93807356 -> 0.93993892 and source6 (40 movies) 0.91246789 -> 0.92593140.
Eight movies regress; the worst is `44b6_9be80b04` at -0.01710537. All source
movies have finite official adjusted scores; no solver fallback or stage cap
occurred. A few movies have fewer than 99 editable problems, not missing output
frames. All 52 were frozen before scoring. Source diagnostic remains distinct
from independently selected/validation cohorts.

The 60-source correction dataset completed (`trajectory-correction-source-v1`):
14,561 components, 301 partial improvements, 107 partial regressions, and 14,153
unlabeled/neutral. Dataset SHA
`e279e24348d17e60e07b8e6bb964ffab95490e502a3f70ac67ecd35928dfa1db`.
Prediction-only feature vectors and cached partial labels are stored separately.

The fixed logistic gate experiment completed (`trajectory-correction-gate-oof-v1`).
Two embryo-held-out fits used the fixed 0.1 L2 regularization and 0.5 decision
threshold, no search. Training for held-out44 used 47 movies/380 labels
(91 negative, 289 positive); held-out6 used 13 movies/28 labels (16 negative,
12 positive). The gate improves source44 slightly (0.94563183 -> 0.94638264)
but worsens source6 (0.92963845 -> 0.92573842). Across all 60 movies, ungated
0.93169308 falls to 0.92819725. Baseline is 0.91803975. Raw edge quality also
regresses versus the anchor; divisions are unchanged. The gate fails its
predeclared source checks and is NOT promoted, fitted on all source data,
evaluated on selection/validation, or included in the candidate. This is
evidence against this particular sparse-label gate, not against all routing.
Thirty-three combined correction/logistic/release/quota tests pass.

### Acceptance output recovery and verified pass

Original watcher 56797 was deliberately stopped at its owned download child,
Python PID45008 (parent31460, watcher1176), after inspection showed it fetching
already available runtime libraries. No GPU job or foreign process was stopped.
Its terminal receipt records the nonzero download exit rather than a false pass.
A filtered CLI retry, session67944, exited on HTTP429 while enumerating library
files at the CLI's default twenty-item pagination. All full validation outputs
were already local and all eight final graphs compared exactly.

`recover-trajectory-event-anchor-smoke-v1.py`, session59753, then completed a
paced recovery: sixty-second initial backoff, 200-item pages, one-second pacing,
HTTP429/5xx backoff, and smoke-only downloads. It recovered 63 files in 54 pages
and rechecked exact remote version1. Receipt
`trajectory-event-anchor-smoke-recovery-v1.json`. All these handles are terminal.
No job rerun, notebook push, competition entry, deletion, or source-model change
was part of the recovery. Future output retrieval must avoid downloading or
rapidly paginating the installed-library tree.

Verifier session41614 completed with **acceptance_passed**. The smoke/full
two-T4/four-worker stages, eight exact initial/intermediate/final graphs,
unchanged cells/protected lineage incidence, no event fallback/cap, and exact
CSV reconstruction all pass. Full wall time 1,086.028519 seconds; simple
199-movie projection 7.504155 hours. This is not a hidden-runtime guarantee.
Actual exact-graph validation score remains 0.94839712 -> 0.96542986.
Acceptance receipt SHA
`2b6b681926f596d422261f34c40cca3d22811a02df691896cdbc68ffef5dfe96`;
terminal SHA `fa4935dd376cc63641993b78152a077c9960a9bbe6b6295d4e718963c31c71d2`.
All original selection/validation quality failures and the separate release
review requirement remain recorded.

### Public-test-only candidate version1 launched

`prepare-trajectory-event-anchor-production-v1.py` staged the same accepted
notebook with its sole executable change `RUN_MODE='production'`. No model,
threshold, worker, dependency, dataset, or gate changed. Embedded contract and
NOTICE retain acceptance-time provenance flags; separate production/release
receipts determine current eligibility. There is no competition authorization
from this engineering test.

Kernel `indarkarhana/biohub-event-anchor-candidate`, version1, was pushed at
17:25:00 UTC by `launch-trajectory-event-anchor-production-v1.py` (session75608,
terminal exit0). Stage `.biohub/staging/biohub-event-anchor-candidate-v1`;
notebook SHA `fa3e4e365d80e0b32c74e5d18384dee2e0d0ef4b84fb8b4d9b74db5248738a35`.
Accepted contract remains
`e8f1c79e6309fb9458c64e9e879b2576d0d570d61856cffafcb64cb50b6a6801`.
Fresh quota was 27.79 hours; pending56231458 reserved12, this one-hour test
reserved2, leaving13.79 beyond outstanding reservations. Antelume had no live
Biohub GPU job; no foreign jobs were stopped. Prior acceptance kernel was freshly
confirmed complete. Earlier submission56231458 still has no score and remains
PENDING. No event candidate has entered the competition.

IMPORTANT: version1 has a **one-hour platform timeout** for this public runtime
test. It must not be blindly submitted to hidden inference. A final competition
version requires the appropriate twelve-hour platform timeout (ten-hour internal
watchdog), fresh quota/outstanding-job accounting, public-output replay/CSV and
runtime verification, and an explicit documented quality-tradeoff release review.
The CLI maps `--timeout` to `session_timeout_seconds`; hidden inheritance is not
assumed safe. The launch receipt explicitly marks `public_test_only=true` and
`final_submission_version_requires_twelve_hour_timeout_review=true`.

## September 14 public replay diagnosis and fresh submission result

The event public test completed all four 100-frame movies in 835.383857 seconds
(13.9 minutes). Actual Kaggle event stages processed all 99 transitions without
fallback or budget exhaustion. The dense movie `6bba_05db0fb1` needed 75.576185
seconds for that stage. The paced evidence downloader completed; installed
libraries were excluded. Version-suffixed source pull returned 403, but the
unsuffixed pull succeeded after authenticated current-version1 verification.

Initial local verifier session53092 failed on the dense event replay. Its
failure receipt is preserved. CPU-only diagnostic session8998 completed:
the original 120-second stage budget processed only 65 transitions and differed
by 851 edges. A 600-second local stage budget (10 seconds per transition)
finished all 99 in 147.016 seconds and reproduced the Kaggle graph exactly,
with zero edge differences and no fallback. No model, candidate graph,
production deadline, or GPU notebook was changed. Evidence is in
`trajectory-event-public-replay-budget-v1.json`. A separate revision2 verifier
was launched to repeat all stages and finish CSV/runtime checks with the
diagnosed CPU-only allowance; actual Kaggle default-budget checks remain.

Authenticated submission inventory at approximately 18:08 UTC reports prior
structured candidate56231458 COMPLETE, score0.946, unchanged from56219125.
GPU quota is27.52 hours remaining. Neither result is the new event candidate.
The completed pending submission no longer needs an outstanding reservation.
Final saved runs and submissions will each conservatively reserve12 notebook
hours with the8-hour floor; no hidden-evaluation quota exemption is assumed.

Fresh score-sorted public notebook listing still includes explicitly named
metric-hack entries; these were not pulled. Clean-family entries include
Harmonic Fusion, the September13 0.947 notebook, Lineage Forge, and LF-DCTTA.
Listing order and titles are not verified clean scores or evidence of gains.

### Final version2 launch and exact public verification

Revision2 verifier session70901 completed in326.61seconds. All four smoke and
four full movies match through all three graph stages; the independently
assembled CSV matches exactly:240,305rows, SHA
`b940d465aafbe62270caaab5dbd8803f6c679781c50763679b4a2a85533ae0df`.
Verification receipt SHA
`8634e199e58327c849ec4204a80076d1ea5743abd94853e38613db9c29e39583`.
Original failed verification remains intact. Runtime projections are7.5042h
from validation,9.3713h from validation-calibrated public work,8.8510h combined,
and11.5445h from the public four-movie makespan alone. Hidden runtime is NOT
guaranteed; the last estimate exceeds the10h internal watchdog and is disclosed.

The final-v2 build records one exploratory-entry decision, not universal model
promotion, a strict-quality pass, or a selected final submission. It preserves
all four cohorts' scores/per-embryo results and all negative movies. The gate
remains excluded. Source-selected coefficients, image models, thresholds,
runtime code and10h watchdog are unchanged. Only notebook disclosure markdown
changes; launch requests the full43,200-second platform cap.

Version2 pushed successfully at18:15:55UTC, launch session31522terminal0.
Notebook SHA`362f3e6288177ef187ddc431022d94a30f64cca10a6142f5b475cdd3f861a1c2`.
Fresh quota27.52h,12h reserved,15.52h beyond reservations. Authenticated current
version2 is private/offline with the competition/runtime input intact. Source
and metadata are downloaded to`trajectory-event-anchor-final-v2-source`.
The watcher session90199/PID30832 is live and observed RUNNING. It downloads
only reproducibility outputs after completion; it never launches or submits.
No Antelume GPU process was live at the launch guard; no foreign process was
modified. The instance remains on/billing, not shut down by this work.

Final verifier/submission entry point is
`scripts/submit-trajectory-event-anchor-final-v2.py`. Before optional`--execute`,
it requires terminal same-contract two-T4 smoke/full output, exact equality
against all independently replayed version1 graphs, graph invariants, exact CSV,
fresh remote version2/source/settings, runtime evidence, fresh quota/full12h
reservation, no pending submission, and fewer than5 entries today. It preserves
the exploratory quality tradeoff and never selects final submissions. It has
been syntax checked but cannot verify before version2 outputs arrive.
Quota/invariant regression suite:35passed.

### Research refresh and next modeling evidence

The September14 latest-run notebook from Bina Salama was downloaded for static
inspection only. It uses the existing UNet/Transformer support-pack family,
`candidate_23_300ep_clean_strict_repair`, detection threshold0.985 and familiar
gap/motion repairs. A fresh run date is not evidence of a new stronger model;
no weights/code from it entered this candidate. Notebook SHA
`bcb1bd98df91a23bdf6afc749a160479a51e8e1bdaab681b4bea5c1e8447c89a`.
[Source notebook](https://www.kaggle.com/code/binasalama/biohub-learned-unet-transformer-ilp-gap-recovery).

Authenticated discussion739685 includes an author's report of roughly0.012
cross-validation motion-link gains but only0.001 public improvement, with a
different local ranking of two public/private notebooks. These are participant
reports, not independently verified experiments or proof of distribution shift.
They reinforce not translating our local gain directly into a promised0.95LB.
[Discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739685).

The September13 nucleus EDA separates local peak intensity from half-max volume,
reports a size-change signal around divisions, and explicitly cautions that
division-rich movie selection and anisotropic boxes limit interpretation. This
suggests testing native-resolution size/shape evolution with same-movie controls;
it does not establish a trained detector gain. No rule from it was adopted.
Notebook SHA`1dd8e153fdb16061f61d107a1357fca0ab1839eac0e51f00a774ad6465d6d9ab`.
[EDA notebook](https://www.kaggle.com/code/zhincez/a-dividing-nucleus-gets-smaller-not-dimmer).

The cited LDA paper's currentv2 abstract reports coarse-grained classification
gains but consistently worse fine-grained CUB results. Any frozen-feature
projection for tracking therefore needs fold-local fitting and direct comparison
with uncompressed features; dimensionality reduction is not a guaranteed cure
for our sparse-label selector. It is not included in version2.
[Paper](https://arxiv.org/abs/2604.03928v2).

A CPU-only all60-source fork16 feasibility audit is now running in session30236,
after the existing two-real-case smoke showed both old incompatible constraints
become feasible. It checks a uniform expanded candidate vocabulary without
inserting ground-truth options, fitting a model, opening selection/validation,
or changing the current submission. This addresses a concrete obstacle to
broader event-model training. Report`trajectory-event-source-vocabulary16-v1.json`;
partial results are not a complete coverage pass or a prepared training dataset.

### Submission56237287 confirmed

Final version2 saved run is COMPLETE. Full public wall802.427643seconds; watcher
90199 completed its filtered download at18:37:47UTC. The one-shot verifier/
submission session60301 finished successfully: all eight smoke/full graph sets
match independently replayed version1 through every stage, graph invariants
hold, and independent240,305-rowCSV reconstruction has the same frozenSHA.
Remote executable cells were also confirmed exactly equal using explicitUTF8
decoding (PowerShell's default ANSI decoding gave a spurious mismatch).

Kaggle authenticated inventory confirms **56237287**, version2, submitted
18:38:48.027UTC, PENDING and no score. This is the third entry today; no final
selection performed. Submission receipt SHA
`1973c5c8e0df52db3a845c0ff6c01a2979c12a3a8049a1cdc30508d3a5787be9`.
Separate confirmation is`trajectory-event-anchor-final-v2-submission-confirmation.json`.
Do not resubmit while pending. Previous56231458 and56219125 remain0.946.

Pre-submitquota27.26h; new hidden run conservatively reserves12h, leaving15.26h
beyond reservation and preserving8h. Revised runtime projections from the actual
final run:9.1349h calibrated public work,8.6991h combined,11.0891h raw public
makespan. Runtime/quality uncertainty remains disclosed, not a0.95 or top-five
claim. Fresh top-five scores were0.970,0.968,0.966,0.966,0.965.

The uniform fork16 regression test confirms the expanded vocabulary is a strict
superset before labels are supplied, retains the incumbent, and makes a distant
two-daughter constraint feasible without changing the graph or inserting
label-dependent options. Candidate/fast-training/final-budget suite16passed.
Full source audit30236 remains live (33/60 last observed, no incompatibility);
the two previously problematic source movies now each cover99 feasible frames.
No new fitted model or inference vocabulary change has been promoted.

## September14 next lane: full source fork16 coverage and learning smoke pass

The source audit30236is TERMINAL success:60movies,5,780prepared transitions,
sixno-known-parent transitions,zero incompatible partial constraints,zero
missing editable supervised frames;1332.828seconds. Scope remains protected:
some individual annotated links are outside editable incidence and are reported.
Full coverage receiptSHA
`9fbc9e46be3cb30383cfa44d397ac2dbd640a885b9a00ac11bcb56b40e952259`.

Real fork16 learning smoke39954completed19.047seconds:1,530,750exact feature
values, two exact serialized case reloads, two real finite optimizer updates.
No coefficients selected/exported. ReceiptSHA
`7aa49cd51459fa4c095e65ac55de7aa1d4539a30dcafcafa36cbb07562d20fb9`.

The separate full corpus build68469/controllerPID23664is live with at most two
local CPU workers. First repaired batches4and5now contain775and768cases with
no incompatibility. As of19:11UTC,sixbatches complete,3and7active. Do not restart.
New source-embryo training/inference and exact-resume control scripts are
prepared, regression suite17passed, but full fits have not launched. Read
`trajectory-event-fork16-v1-findings.md`for the prospective contract and next
commands. Submitted candidate56237287is untouched; no additional submission.

### 19:16 UTC: new source fits genuinely running

Corpus68469 completed all5,780cases /60movies /3,740,379,133bytes in1449.203s.
Real end-to-end resume audit9284 passed both embryos exactly (2+2updates vs4
uninterrupted), including optimizer/RNG/schedule state, in22.109s. Full fits
resumed fromstep4 after fresh proof/hash checks:
source44session80955/PID7572,1,223cases per epoch; source6session78731/PID43524,
4,557cases per epoch. Both have saved their first50-update checkpoints and remain
live. Fixed3epochs,2h invocation caps, checkpoint every50updates, local CPU only.
No final model, held-embryo quality gain, or extra submission is claimed.
Read`trajectory-event-fork16-v1-findings.md`and the full-fit launch receipt.
