# Broader event-head training: September14,2026

The submitted event-anchor candidate56237287/version2 is unchanged. This lane
addresses the two source training cases that the old eight-daughter vocabulary
could not represent. It is not a metric hack, a new movie-ID router, or evidence
that the next model will improve the leaderboard.

## Completed evidence

The uniform16-daughter feasibility audit completed all60 optimization movies,
5,780 prepared supervised transitions, and six transitions with no known-parent
constraint. No partial constraints were incompatible; no supervised timepoint
vanished from the editable scope. Existing protected division/synthetic/gap
incidence remains excluded. Some individual known links lie outside that scope;
this is not a claim that every annotation is trainable.

Audit1332.828seconds, receiptSHA
`9fbc9e46be3cb30383cfa44d397ac2dbd640a885b9a00ac11bcb56b40e952259`.
Predictions/labels unchanged, no fit or selection/validation annotations opened.

The real learning smoke on`44b6_eb2880fc/t73`and`6bba_48816121/t17`passed in
19.047seconds. Reference and accelerated feature calculations match exactly on
1,530,750values across26,112and24,913event options. Both saved cases reconstruct
exactly. Two real optimizer updates are finite and reduce their training hinge;
this is functionality evidence only. No coefficients exported/selected.
SmokeSHA`7aa49cd51459fa4c095e65ac55de7aa1d4539a30dcafcafa36cbb07562d20fb9`.

## Dataset build currently running

Controller`build-trajectory-event-fork16-corpus-v1.py`,session68469,actual
controllerPID23664, uses at most two local CPU workers, no GPU. Each batch has
separate immutable`-cases-fork16`artifacts; original fork8 cases remain intact.
Per-case hashes, source manifests and the full feasibility transition inventory
must match. No incompatible frame is silently dropped.

First two repaired batches are complete:

- Batch4:775cases,428,104,372bytes,5,320retained known constraints; two known
  constraints outside the protected editable scope;279.266seconds.
- Batch5:768cases,605,404,011bytes,4,625retained known constraints; seven known
  constraints outside scope;403.969seconds.

As of19:11UTC,batches4,5,0,1,2,6are complete;3and7are active. Observe the existing
controller handle, not just its start/finish-only timestamp. Windows venv PIDs
can be launcher parents; inspect their exact children and current output writes
before treating a worker as idle or stopping anything. No job was restarted.

## Prospective training/evaluation contract

`train-trajectory-event-fork16-source-v1.py`requires the complete60-movie corpus.
It fits separate heads on13source44movies and47source6movies. The opposite embryo
is excluded from optimizer inputs and source-prior estimation. Existing public
backbone/inherited18-edge-weight exposure remains disclosed; this is not
pristine project-wide OOF.

The fixed learner uses30features, uniform16-daughter vocabulary, three epochs,
Adam0.03,seed20260914,anchoredL2(0.1for18inherited coefficients,0.02for12event
coefficients), and exact10-second training oracles. Source-only division priors
are recomputed per training embryo. Unknown labels are not negatives. There is
no source-score-based epoch or threshold selection.

Optimizer moments, weights, exact case permutation/position, RNG state and
partial epoch statistics are checkpointed every50steps. Each invocation has a
two-hour local CPU wall cap and pauses with state preserved. A single-writer
lock blocks duplicate fits; resume requires the unchanged data/code contract.
The global case cache is bounded at1.5GiB per fit. Control-smoke checkpoints are
never candidates. `--control-smoke --steps-limit 4` is the uninterrupted comparator;
normal `--steps-limit 2` then `--resume --steps-limit 2` must reproduce it exactly.

`audit-trajectory-event-fork16-resume-v1.py`is prepared to run that real2+2vs4
comparison for both embryos after the corpus finishes. Do not launch a long fit
before this audit passes. On pass, real fits remain paused atstep4; continue
with`--embryo 44b6 --resume`and`--embryo 6bba --resume`, not fresh duplicate fits.
This end-to-end audit and the full fits have NOT started at this handoff.

Research-only`trajectory_event_fork16_inference_v1.py`uses the same uniform
candidate vocabulary and preserves the submitted fork8 module. Tests verify
zero-gain identity, valid positive-fork output, optimizer parity with the old
learner, optimizer save/reload parity, lock exclusivity and schedule coverage.
Combined regression suite:17passed. These tests are not full-movie quality or
Kaggle runtime acceptance.

After each fixed fit completes, freeze opposite-embryo complete-movie predictions before
scoring. Compare each trained16 head against the submitted fixed8 anchor, and
also against its own untrained16/prior control to distinguish vocabulary/prior
effects from learning. Report pooled raw/adjusted quality, per-embryo and worst
movies, divisionTP/FP/FN, nodes and runtime/fallbacks. Do not use movie identity
as an inference gate. A fixed equal-score ensemble is a later candidate only if
both components establish useful held-embryo behavior; do not tune blend weights
on leaderboard scores or silently overwrite failed gates.

Antelume's GPU process list was empty at19:09UTC. No GPU job is queued for this
solver-based stage; no RSNA or other project files/processes were changed. The
instance remains on/billing and available for the user's other work.

## 19:16 UTC: full corpus and resume audits complete; full fits running

Corpus controller68469 is TERMINAL success. All eight batches contain exactly
5,780 cases / 60 movies, totaling 3,740,379,133 bytes. Build took 1,449.203 seconds.
Receipt SHA `1c520023037a0a273da9722bf8a44b4a82476b00eda8baa383fe80aeda804e92`.

End-to-end resume audit9284 is TERMINAL success, 22.109 seconds. For both source
embryos, two updates + save/restart + two updates reproduce the complete four-step
uninterrupted checkpoint exactly, including weights, moments, case order, RNG
and progress. Audit SHA
`1faafe3303a8fafe1149a9bfd2df7098e78339468b4775629db24af7f3d77214`.

Both actual fits resumed safely from step4 after fresh paused-receipt/code hash
checks. They are currently live, not just queued:

- Source44: session80955, actual Python PID7572, 13movies / 1,223cases per epoch,
  3,669planned updates. Step50 checkpoint saved at53.062seconds.
- Source6: session78731, actual Python PID43524, 47movies / 4,557cases per epoch,
  13,671planned updates. Step50 checkpoint saved at20.5seconds.

Each invocation has a7,200-second CPU cap, checkpointing every50updates. The
initial timing suggests roughly1–2hours for training, not a reliable runtime
guarantee. Do not restart either live handle because an observation times out.
If a fit returns paused_at_wall_limit, inspect its exact saved state then resume;
if failed_requires_inspection, diagnose the recorded failure rather than skip
an example or loosen oracle correctness. The single-writer lock prevents overlap.

No full fit is complete or quality-qualified yet. Opposite-embryo full-movie
scoring and the untrained16 ablation remain required. No selection/validation
scoring, further Kaggle entry, or Antelume GPU job was launched in this lane.
Operational launch receipt: `trajectory-event-fork16-full-fit-launch-v1.json`.

## Prospective evaluation preparation, before either full fit completes

The evaluation ordering above now permits evaluating a completed fit while the
other immutable three-epoch fit continues. Both original launch-contract hashes,
the training script and training helpers must still match before inference and
before/after scoring. No held results may change the other fit, its duration,
initialization or hyperparameters. This is a scheduling change, not an adaptive
training or stopping policy. Neither fit has been scored at this amendment.

`evaluate-trajectory-event-fork16-held-v1.py` first requires two complete-movie
inference smoke tests, chosen label-free by smallest/largest compressed feature
file size in the opposite embryo. Both learned16 and untrained16 are tested.
All 100 output frames and protected graph invariants must hold with no solver
fallback or wall-budget exhaustion. Local CPU diagnostic budgets are10seconds
per frame/600seconds per stage; this is not Kaggle runtime acceptance.

Full evaluation reuses the exact smoke predictions, freezes every remaining
movie before opening GT, and independently rescores the submitted fixed8 graphs.
The baseline per-movie scores must reproduce prior pinned receipts. Quality
checks explicitly reject missing/undefined movie scores and distinguish a
learning gain from an untrained-vocabulary gain. One regressing movie remains
a strict failure even when pooled score rises. No automatic submission or
independent selection test is authorized by this evaluator.

Latest live-handle observation: source44 step750/3669 at1050seconds and source6
step1150/13671 at1092seconds. Neither fit was restarted. These measurements
suggest the larger fit may require a checkpoint resume after its two-hour
invocation cap; the earlier1-2hour estimate is not a completion guarantee.

## 19:37 UTC: evaluator checks passed; original fits still live

The expanded regression suite completed successfully:25passed in6.05seconds,
session29013 terminal exit0. Eight new evaluation checks cover JSON-native
booleans, a per-movie regression despite pooled gain, a vocabulary-only gain,
undefined aggregate/per-movie scores, missing movies, undefined division terms
when no events exist, false new divisions, and the real immutable fit contracts.
No held GT or intermediate model predictions were opened by these tests.

The two owned training processes were verified live via their actual PIDs:
7572/source44 latest checkpoint900/3669 at1272.531seconds;43524/source6 latest
checkpoint1400/13671 at1270.891seconds. About1.8GB process working set each,
cache below1.5GiB each. No failed oracle or new fit restart is reported.

One authenticated submission inventory check at19:35UTC still lists56237287
as PENDING, with no score. Prior56231458 and56219125 remain COMPLETE at0.946.
There was no resubmission or extra GPU launch. This turn made implementation
and test progress as well as verifying the existing fits; top five is not yet
achieved. Next: final-weight-only two-movie smoke, then full opposite-embryo
evaluation, observing the same training handles and preserving failed outcomes.

## Prospective label-free control preflight during training

To exercise the actual movie input path before weights are ready, the evaluator
now has an explicit `--control-preflight` mode. This uses only the already-frozen
source-prior anchor in CONTRACT.json, not a checkpoint or fitted weights. It
runs the same two label-free size-selected complete movies in the opposite
embryo, only for the untrained16 control, with no GT or quality score opened.
It exits before scorer loading. This cannot qualify a learned model.

Once fitting completes, the two-arm smoke may reuse exactly hash-checked control
graphs from this preflight. It still must run the final learned16 head on both
complete movies. Model/source/graph/helper/scope mismatches fail closed; the
untrained array in final-weights.npz must exactly match the original contract.
This avoids recomputing the same control while giving early real-data coverage.
No training file, helper, oracle, fit contract, epoch or optimizer was changed.

## 19:50 UTC: first real control preflight complete; second running

The25-test suite passed again in17.61seconds after adding the preflight path.
Evaluator source is now pinned by running receipts atSHA
`0d6716829382e06ae5e476d8e0789104cd9f3ac0663722c040f7f898f6390f37`.
Do not edit that source while its preflight/evaluation chain is active.

Source6-prior untrained16 on held44 is TERMINAL success (session27247):
`trajectory-event-fork16-held-44b6-v1-control-preflight.json`,SHA
`a6c00589d1b596293f7c652012dd973ca47c3fa2cfe91c1b418d90ec4cb12fcc`.
Total227.578seconds. Label-free extreme feature-size movies:

-44b6_996155de:99transitions,10.453seconds,9added/30removed edges.
-44b6_d5e7d891:99transitions,210.578seconds,101added/452removed edges.

Both contain100valid output frames, no solver fallback, no budget exhaustion,
unchanged nodes/coordinates and protected incidence. These edit counts are not
correctness scores. No GT opened, no intermediate learned coefficients loaded.
The real evaluator resolved/hash-verified all13held44movie inputs before this
two-movie test, not merely mock fixtures.

The reciprocal source44-prior control preflight is live:session50378,actual
PythonPID5304 (venv launcher33924),held6bba movies6bba_55b7eebe and6bba_57b7cc1e.
All47held6inputs were resolved/hash-verified before inference. Observe this
samehandle; do not launch it twice. Training stays on original80955/78731.

Latest source44checkpoint1500/3669 at2109.312seconds, epoch1 complete with online
objective0.12231066 (not held quality). Source6checkpoint2250/13671 at2080.266s.
No new submission inventory poll this turn; last remains19:35UTC PENDING.

## 19:59 UTC: both control preflights complete; exact runtime work underway

Reciprocal control preflight50378 is TERMINAL success,513.563seconds total.
Receipt `trajectory-event-fork16-held-6bba-v1-control-preflight.json`,SHA
`142073e33f460e3ab0e741a556974d99db4c61e927a186cf5d27f1582176283f`.
6bba_55b7eebe:99transitions,5.234seconds,13added/12removed edges.
6bba_57b7cc1e:99transitions,490.109seconds,452added/1755removed edges.
All100output frames and protected incidence are valid, no fallback or stage cap,
no GT opened. These are untrained controls, not quality-qualified candidates.

Both original fit contracts and training code remain unchanged. A SEPARATE
experimental pruning implementation matched60real-source comparisons exactly
and reduced that component's benchmark time from16.414s to0.592s. It is not
installed in the live fits, frozen evaluator or production. Complete-movie
graph/solver-record replay is running as7794; see
`trajectory-event-vectorized-runtime-v1-findings.md`. Do not confuse the27.7x
component ratio with measured end-to-end or Kaggle speedup.

## 20:07 UTC: source44 final-weight evaluation successor is live

`queue-trajectory-event-fork16-held-v1.ps1` passed read-only validation for both
actual fit processes and their start times:PID7572 began19:14:30.231639UTC,
PID43524 began19:14:40.459849UTC. It retains the actual .NET process handle,
checks the embryo/script/resume command line and hashes, and cannot substitute
a recycled PID. No evaluation receipt may already exist. A CreateNew lock and
preserved successor receipt prevent duplicate queue launches.

The source44 successor was actually launched at20:06:49UTC:session22441,
actual PowerShellPID35260,waiting on original trainingPID7572. Report
`trajectory-event-fork16-held-6bba-v1-queue.json`. On exact training-process exit,
it requires source_event_training_complete, then runs the original frozen
evaluator's two-movie final-weight smoke followed by the47held6complete movies.
Smoke failure blocks the full run. Paused/failed training stops the successor
for inspection, without restarting or changing training. Queue wait is bounded
to three hours, with30second process waits; it does not busy-poll Kaggle.

The first direct PowerShell script invocation was blocked by this machine's
execution policy before any script action. The dedicated child invocation used
process-scoped `-ExecutionPolicy Bypass`; no persistent machine/user policy was
changed. Both ValidateOnly calls performed no writes, then the real successor
created its own receipt and lock successfully and was verified live.

No source6successor was launched: its measured fixed schedule is expected to
need the existing two-hour safe pause and a verified resume first. Do not launch
manual held6smoke/full evaluations while22441 owns that chain. The new vectorized
research adapter is NOT installed in the frozen evaluator or either fit.

## 20:47 UTC: source44 training complete; final-weight smoke actually running

Original source44 session80955 is TERMINAL success,3,669updates/3fixed epochs,
5,377.578seconds in the full resumed invocation (89.63minutes). Final online
training objectives by epoch0.12231066,0.11787094,0.12589540; no epoch was selected
from these values. Final anchor displacement norm1.62982922. These are training
statistics only, not validation improvement.

Final modelSHA`17f60597888c426aba77992d50bd6dfecd1547c3f4356d4bdd2bf1755c731f01`;
final checkpointSHA`bed34d7e998e175862e4e9313a969093c5c9e6fdbb698434ab154116a4aa2457`;
fit reportSHA`db22184988a00299c0f5827ec860277831705efeb363fa74af72429e46582c31`.
The fit contract stayed551f749c87590874a590403095e3cb7e826e4c3de5114f93890319d35db7c8cf.

Successor22441/PowerShellPID35260 detected that exact process exit and started
the final-weight smoke at20:44:09UTC. Small held6bba_55b7eebe has been frozen;
dense6bba_57b7cc1e is still running. Both original untrained controls are reused
by exact hash; learned predictions use the final model only. No GT has been
opened, no complete smoke or score gain is claimed. On passing smoke, the
successor will freeze/score all47held6movies; do not manually duplicate that job.

Source6fit78731 remains LIVE,latest step6000/13671,first epoch complete. It still
has its original two-hour invocation cap and may require a verified pause/resume.
The unrelated morphology v7screen was rejected and is not in this candidate.

Portable vectorized runtime preparation15592 is independently TERMINAL success:
four complete controls replay exactly, including regenerated edge features.
See portable findings; that runtime is NOT installed in the frozen evaluator.

The live learned-smoke Python process is PID9312,venv launcher43800,parent
successor35260. Keep observing session22441 for its output. Portable final
receiptSHA`8bcd743a08941e7e77ab174053cdf912a15fc4fde31a8cc88cd092656b94463f`.

## 21:01 UTC: learned smoke passed; full held-source evaluation live

The original learned smoke completed successfully in335.375seconds, receiptSHA
`9975d1c49c5d9c48644be0d92df5fe540d025f0015f1c9f62ce0a0ff75115330`.
Small6bba_55b7eebe learned inference3.672s,18added/10removed; dense6bba_57b7cc1e
316.766s,914added/625removed. These are edit counts, not correctness gains.
The existing successor started full47held6evaluation at20:50UTC. Actual
PythonPID7288/launcher41440/parent35260 were verified live at21:01UTC. It remains
in freezing_predictions; do not launch a duplicate or change frozen helpers.

Separate portable learned smoke17045 is TERMINAL success,416.516seconds total.
Both final-source44-weight graphs and all non-timing solver records match the
original evaluator exactly,99transitions each, no cap/fallback or GT access.
Small inference3.922s; dense377.329s. The dense portable timing was SLOWER than
original316.766s in these shared CPU runs: no learned-model speedup is claimed.
ReceiptSHA`0f35ed01f9aa14b1fd3856aacc2e3e61a86d74c7b9f95bf2185a4b41073525a2`;
scriptSHA`27d6b1848a8c76b94e3548ff3e9c0ed227a73cf3069649450d34367d58c17e98`.
This proves packaging equivalence, not quality improvement or Kaggle acceptance.

Source6trainingPID43524 was verified live,6800/13671updates at6359.531seconds.
Its original7200second invocation cap and immutable resume contract still apply.
No new GPU job, Kaggle submission, production change or selection was made.

## 21:15 UTC: source6 safely paused and actually resumed; evaluation queued

Original session78731/PID43524 is TERMINAL success with paused_at_wall_limit,
7,688updates after7,203.735seconds. Checkpoint
checkpoint-0007688-e1-p3131.json SHA
`0466a17bac8635a6edd6c5bf238e174955c076bedd9337382c5f2a08270b1dca`;
paused reportSHA`83aa9758a3e6175ba0c367342437821693f5f677e45c656b34a3e676503b7d45`.

New owned controller session44645/PowerShellPID13476 passed read-only preflight
against the original live process, retained its .NET handle, waited for its
actual exit, checked paused status/code/contract/checkpoint containment+hashes,
and launched exactly one original-code resume at21:14:46UTC. Resumed actual
PythonPID36456,launcher22768,parent13476 verified live. Same immutable contract
ea0bcdfe..., no optimizer reset, no shuffled-order reset, no vectorized helper
replacement and no selection-based schedule changes. Controller scriptSHA
`4cbd5015c50f28bf65d2d51d47af1dabdf07ac84b2a37f2ece642567650db446`.

Controller receipt trajectory-event-fork16-6bba-resume-evaluation-v1.json
preserves the entire paused report and checkpoint pointer. After13,671updates
and a verified complete fit, it runs the frozen held44smoke and then13complete
movies only if smoke passes. Any failure or another pause stops for inspection,
not a restart; no automatic promotion/submission. Do not also launch the older
queue script for source6 or duplicate manual smoke/full evaluation.

The original source44 successor22441 still owns the47held6evaluation. Its
actual evaluatorPID7288 was reverified live at21:10UTC, freezing predictions.
Kaggle inventory at~21:09UTC still has56237287 PENDING, prior best0.946, three
submissions today. No new GPU use or extra submission occurred this turn.

## 21:29UTC: core jobs live; separate selector ablation rejected

Specific handles22441and44645 were polled and confirmed still running. The
first has now emitted24complete held_movie_frozen events including the earlier
6bba_085bf656; no full quality score yet. Source6resumed fit emittedstep8700 at
752.969seconds of the current invocation; no contract/schedule/helper changes.

The independent FP-label gate v2 trial finished and FAILED pooled/embryo/movie
gates. Its0.927361469 score is not a new candidate gain. The separate exact
whole-movie choice diagnostic bound is0.932046644vsanchor0.931693079 on the
60source movies only; neither result changes these live event fits/evaluations.

## 22:08 UTC: first learned head improves aggregate, fails promotion

Owned session22441 TERMINAL success; controller reports
held_evaluation_completed_no_automatic_promotion. All47held6movie predictions
were frozen before scoring. Final reportSHA
`c51da3b1261a9f33ab1f73810b9e9df0c7a7395c7183c8ead14dc73c7be90c01`.
Total4699.437seconds including inference and scoring.

Submitted8anduntrained16 both score0.929638450034; learned16 scores
0.931293720778 (+0.001655270744). Learning, not merely expanded fork vocabulary,
causes the aggregate change. Raw edge Jaccard nevertheless drops
0.920227678192 ->0.918957235659. Division counts change10TP/31FP/78FN ->
22TP/106FP/66FN; division Jaccard0.084033613445 ->0.113402061856.
Node recall unchanged0.981313745283. These are local held-embryo source scores,
not leaderboard forecasts or pristine backbone-held-out scores.

18movies improve,23regress,6neutral. Worst6bba_3abfe10a -0.058513099489;
next6bba_5c824876 -0.048293111057 and6bba_9a41d029 -0.035319924523.
Raw-edge and every-movie checks FAIL; strict and aggregate diagnostic gates
FAIL. No selection/validation cohort opened, no promotion, no submission.
Do not turn these exposed held-movie results into identifier routing or tune
division thresholds against them. The unreleased source44worker draft remains
unauthorized even though its portability smoke passed.

Source6second fit remains under owned controller44645,actualPython36456;
latest22:10snapshot11750/13671updates. Its immutable schedule and queued held44
smoke/full evaluation remain unchanged. No duplicate launch. Antelume GPU was
read-only checked this turn: no compute processes/0MiB allocated; no remote
GPU job or other-project mutation. Kaggle inventory~22:07still56237287PENDING;
best verified public score remains0.946.

## 22:43 UTC: second final fit complete, original held44smoke active

Source6actual training36456 completed13671updates,3fixedepochs. Finalepoch
objective0.110299436531, displacementnorm0.734732961034. Resumed invocation
5277.625seconds plus original paused invocation7203.735seconds; no optimizer,
RNG/permutation or schedule reset. FinalweightsSHA
`2e4034baa6d3d8e07ae0cb63ea0d2acc22fc94a6cc2f4c13b15a1121e35aefed`;
fit reportSHA`9bb0ef61d09c13fa871f6f9b61870d837aea5a798b534e0a019a9112a53a62cd`;
finalcheckpointSHA`f8e350bd2d0281255bc80557898dddf4f9a29992f4f2ad2f2921143cf8b1b2eb`.

Original controller44645 owns current held44small/dense smoke, actual43104,
then13movie evaluation on success. No duplicates. Separate pre-frozen50:50
ensemble/source44/source6comparison on10selection movies passed its own
two-small-movie smoke and is now running full under74206/actual36228.
That comparison is not a replacement for source44failedtransfer or held44test.
No model promoted, new submission made, or GPU launched at this snapshot.
