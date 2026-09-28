# Submission readiness — September 10, 2026 UTC

Target: a clean, reproducible submission competitive with the best legitimate
public approaches; user target 0.945. No result below establishes that target
or a new competition submission. Public leaderboard superiority remains unverified.

## Latest execution update

22:51 UTC: Antelume fullpairedD4 diagnostic and CPU rescore COMPLETE. Pooled
score0.944839→0.947619, rawedgeJ improves, but both6bba movies regress; frozen
embryo/movie gate FAILS. No promotion/submission,0/5; not a Kaggle0.947619 score.
All8outputs/logs/rawcandidates locally backed up and hash-verified. GPU job
932.901s; GPU free since22:43–22:44, instanceleftup/sharedRSNAuntouched.
CPU stageattribution: correctedrawdetector matches1418/1418annotatedcells,
with laterILP/postprocessinglosses. Next CPU work targets those losses, not a
larger detector or publicreplica. No positiveGTdivisions in this small diagnostic
subset and publictrainingoverlap; independentgeneralizationgate remains unmet.
See public-d4-full-movie-v1-result.md. Older running entries are historical.

22:30 UTC: Antelume now RUNNING after user confirmation. Real D4 GPU preflight
passed21.614s; full-pipeline eight-frame paired smoke passed28.496s. Four complete
movies downloaded/hash-verified; sequential eight-arm full diagnostic RUNNING,
session29029, one-hour hard cap. Same public weights/settings, only D4 geometry
corrected.21focusedtests pass. No tracking scores yet; public-training overlap
explicit, no independent-promotion claim,0/5. Kaggle GPU unused and RSNA untouched.
See public-d4-antelume-v1-execution.md. Earlier stopped/staged entries are historical.

22:00 UTC update: public-baseline enhancement is now the main lane. Found and
fixed a concrete seven-unique-views-in-eight-passes bug in the recent LF-DCTTA
family. Kaggle metadata reports 0.947 for the fixed public reference, NOT ours.
Source geometry audit and ten tests pass. Exact 65.7 MB, image-only paired GPU
preflight staged with 20-minute cap; no GPU launched. Antelume's fresh read-only
check reports STOPPED. User asked to free one GPU; RSNA untouched. See
public-frontier-gap-20260910.md and public-d4-preflight-v1-design.md.
Tree quality and independent verification are COMPLETE: NLL0.345773,
9848 correct parents,83 absent versus required115; REJECTED. History-head
optimizer smoke passed, but further standalone history fitting is paused.
No qualified submission or new measured tracking gain;0/5. Old updates below
are historical, not live-job descriptions.

21:15 UTC observation: tree quality21863 confirmedlive with ninefolds saved;
full12fold gate/verification still pending. Separate preceding-image-flow
feature audit COMPLETE43.125CPU s: all1188 original fitting packets/IDs/geometry
verified, history available on1176 noninitial transitions, no new extraction.
Real1,378,107choice smoke independentformulaerror0, exact32/17block equivalence,
no labels loaded.41combinedtests pass5.82s. This is data/functionality evidence,
not another trained model or score gain. See focus-pair-history-v1-result.md.
NoGPU/cloud/RSNA/submission;0/5. Run the prepared direct-worker verifier only
after all12treefoldscomplete; leave current scientific files unchanged.

20:56 UTC: quality21863 still live, fivefoldscomplete/sixthrunning. No pooled
gate yet. Native label-free inference passes all1,378,107choices, including
1167unknown targets: originalsmoke5.062s vsportable15.063s; actualfull-trained
model5.907s vs13.109s, exactoutputs. Fitting-manifest workload302,977,426all
choices vs4,691,320labeled choices, so fullinference budgeting remains needed.
Actual first-fold native verifier smoke passed231,230choices;30tests pass.
Full verifier/controller prepared but not started. Use run-focus-tree-verification.py
after all12complete to monitor actualPID, not the Windows launcher. No GPU/
cloud/RSNA/newsubmission. See focus-pair-tree-inference-v1-result.md.0/5.

20:32 UTC: first full nonlinear100tree fitting model recovered and replayed,
actualpeak4.88GiB,244.203s recovery, no candidate/data/settings change. Complete
12fold quality screen now LIVE21863, expected60-90minCPU with7200s hard cap,
900s/fold,6GiB actual-worker guard and7GiB free-memory prelaunch check.
First model reused: onefold417parents/11absent, NLL0.548993; not a promotion.
13finalfocusedtests passed1.20s. No GPU/cloud/RSNA/submission.0/5. See
focus-pair-tree-progress-20260910.md for exact recovery/model receipts.

20:23 UTC: profile87429 was stopped after its actual numerical child exceeded
6GiB; original guard watched the Windows venv launcher, not the worker. Only
verified owned child30456 was stopped.75tree checkpoint preserved. Corrected
direct-interpreter guard passed a real64MiB allocation test; actual workerPID
now confirmed. Resume75+25 exactly reproduces original100tree real smoke.
Full first-fold recovery36363 is LIVE, same settings/candidates, redundant
source arrays released before fitting; about4GB observed,900s/6GiB guard.
No quality/submission claim and no GPU/cloud/RSNA change.0/5.

20:15 UTC: full72D recovery completed12folds and independently verified.
NLL0.349453,parent9854/10754,absent84/161. LDA has lower pooledNLL0.342701
and9880parents, same84absent. Both fail115requirement; neither promoted.
No new submission. Fixed nonlinear candidate-ranking residual passed the
original real20group smoke and18tests; now one11movie fitting-only time/RAM
profile is live87429,900s/6GiB cap. No held-out quality score yet, no GPU/cloud
or RSNA mutation. See focus-pair-appearance-comparison-v1-result.md.0/5.

19:49 UTC: old full-feature run2266 is TERMINAL: firstfold hit500iterations,
no held-out score or model. Exact curvature audit supports ill-conditioning;
bound-preserving optimizer coordinates reduce condition30274.7->1.85 and
pass original real-smoke objective/decision checks. New CPU recovery5342 is
live, same objective/model/max500/quality gates,12folds,9000s cap. Checkpoints
every25iterations, including failure states. Bounded label-free inference also
passes on1,378,107 real fitting-packet choices, no filtering/coordinate change.
No new GPU/cloud/RSNA/diagnostic/source/target/submission. See
focus-pair-appearance-recovery-20260910.md. Goalunfinished,0/5qualified.

19:31 UTC: LDA12folds COMPLETE and host-verified: NLL0.342701,parent9880,
absent84/161; all12movie NLLwins. Missing-parent requirement115 FAIL, so no
promotion/finalfit/submission. CPU421s plus145.125s replay. Full-feature arm
continues, session2266 confirmed live at firstfold100iterations/121.125fit s.
Verifier39992 terminal. No additional GPU/cloud/RSNA mutation. See
focus-pair-appearance-lomo-v1-lda-result.md. Goal remainsunfinished0/5.

19:26 UTC: richer full/LDA appearance comparison is now genuinely running on
CPU in session2266, LDA then full,12folds each,9000s hard cap. Eight LDA folds
saved at latest read; no pooled gate or qualified candidate yet. Complete
descriptors/provenance were replayed; GPU equivalence smoke v1/1 COMPLETE and
host-verified (9.041s launcher). Exact CPU caching reduces objective calls from
8-10s to0.7-0.9s full/0.3-0.5s LDA;46 tests pass. No further GPU needed for this
comparison. Antelume STOPPED at19:17; Kaggle8.22h then, refresh before future
launch. RSNA/shared state untouched. Full live rules and code requirements
unchanged; targeted new discussion review completed. See
focus-pair-appearance-v1-progress-20260910.md. Goal unfinished;0/5 qualified.

18:33 UTC: fixed class-weighted candidate ranker completed and verified in
597.297 CPU-only seconds. Parent 9891/10754, absent 73/161, NLL 0.362457;
11/12 movie NLL wins versus original neural. Absent requirement 115 still
fails, so no final fit, diagnostic/source/target use or submission. All workers
terminal;21 tests pass. Separate exact fitting error attribution rules out
combining the previous ranking with an unchanged physical rejection veto:
it would retain only 9550 correct parents versus 9835 required. No GPU used.
User-supplied LDA paper reviewed; a richer pair-feature CPU comparison is worth
testing, but no LDA model has run. Latest public-source refresh detected an
explicit negative-time synthetic graph exploit in dhiaalhemdani's updated
notebook; now excluded before future pulls. See the balanced-ranker result,
paper assessment and public-refresh-20260910-1831.md. Goal unfinished;0/5.

18:08 UTC: full-candidate ranker completed and verified. Correctparents9963vs
neural9835;NLL0.315919vs0.667407,all12movie NLLwins. But absent28/161vsphysical115
fails frozen acceptance. No finalfit/diagnostic/source/target/submission.
All4,691,320 candidate/null rows for10,915 targets preserved; data+real smoke
87.938CPU s,LOMO254.656s. Saved fold decisions/objectives/gates replayed;17tests
pass1.23s. All processes terminal41809/71461/4475. See focus-candidate-ranker-v1-result.md.
AWS credentials now WORK: user refresh17:46UTC, EC2 query17:57UTC confirms
Antelume STOPPED,g5.xlarge,no publicIP. OldExpiredToken blocker resolved.
No GPU, instance action or RSNA/shared-environment mutation. LastKagglequota
8.80h at17:18,refreshbeforelaunch. Rankinggain is real on this correction screen,
but abstention tradeoff rejects the candidate. Goal unfinished;0/5qualified.

17:43 UTC: conditional-variance model passed14movie fitting-only screen
(NLL5.741790->5.698241,9/14wins,mean/MSEunchanged), then FAILED complete-source
tracking. Score0.7761649944vsconstant0.7838611993;true divisions4vs5; all8movies
worse adjustededge score. Original67ebd073 per-movie guard alsofails(-0.044650).
Actual all14fits/finalmodel and all168586rawnodes/139654candidateedges verified;
parent/FOCUS-flow/constant controls and official aggregation replayed exactly.
No promotion/retuning/submission. See focus-conditional-variance-v1-result.md.
CPU screen1.969s/source94.125s;23tests1.89s pass; all processes terminal.
NoGPU/newtarget/diagnostic use; Antelume STS againExpiredToken; RSNA untouched.
Next research should directly supervise candidate ranking, not more presence-
only or covariance-only sweeps. Lastquota8.80h at17:18,refreshbeforelaunch.
Goal unfinished,0/5qualified; this turn produced full-source experiment evidence.

17:30 UTC: CPU quadratic-presence12movie LOMO screen completed and independently
verified; all five gates FAIL. Pooled NLL0.656424vslinear0.633682, parent9889vs9894,
absent34vs33(original72). Only4/12movie NLL gains; no diagnostic/source/target
opened and no final model fitted. Actual fitting-only error budget shows86.727%
of originalNLL is conditional parent ranking; presence-only correction ceiling
9909/10754 correctparents. Retire presence-only calibration for this fixed
representation; next research must change parent ranking or detector/features.
See focus-quadratic-presence-v1-result.md. Twenty-one tests pass3.81s.
No GPU used/no live worker. NamedAWSsession ExpiredToken, defaultNoCredentials;
no environment fallback or credentialfile refresh. RSNA/cloud untouched.
Goal remains unfinished,0/5qualified; this turn produced concrete CPU evidence.

17:20 UTC: exact-GT dropout training v1/1 COMPLETE and HOST-VERIFIED. Final
NLL 0.221171, parent 2577/2645, absent 11/27: all three unchanged gates FAIL.
All 800 updates/nine checkpoints recovered and verified; no source/target
evaluation or submission. Worker 122.092s, launcher 203.752s. See
focus-gt-parent-dropout-training-v1-result.md; host receipt e5995108... .
All local checks pass (35 tests, 4.56s; syntax and tracked diff checks clean).
No Biohub GPU job active. Follower20881/download12261/verifier26593 terminal.
Kaggle remaining 8.80h at 17:18 UTC; 8h reserve intact. AWS STS confirms
ExpiredToken for named Antelume profile; request to refresh session sent.
Current cloud utilization unknown; RSNA/instance unchanged. Request 0/5 new
qualified submissions. Do not extend this failed configuration unchanged or
submit it. Fresh credentials alone do not establish model quality.

17:14 UTC: exact-GT smoke completed and host-verified; four updates and exact
checkpoint reload passed. Full 800-step exact-GT training v1/1 is now launched,
sole log follower 20881. Fresh quota 8.86h; 900-second cap preserves 8.61h.
Stage notebook 04614e125862182fa0c9a6f4659773338a48b6c577e4f3e65805c31cb5fafed7.
Eight current tests pass. No quality result yet, no source/target expansion,
no new submission. Antelume EC2 query again RequestExpired; cloud state unknown,
no RSNA/cloud mutations. New skomuro public source inspected only; explicitly
leaderboard-configured, no adoption/certification. See public-refresh-20260910-1714.md.

16:58 UTC: fitting-only difficulty analysis identified overly easy14um dropout
examples:99.91% physically solved vs71.43% natural nulls. Exact-GT7um alternative
now audited on12fit movies and all1,188 original label sets replayed. All1,121
serialized augmentations independently replayed;31 physically wrong-parent cases
vs old1, mean nullNLL0.076444vs0.004479. Data feasibility PASS, but still97.23%
physically solved: no model-quality claim. No new GPU or submission this turn.
No live Biohub job; audit36912/verifier21069terminal. Eleven tests pass2.83s.
See focus-null-difficulty-v1-result.md and focus-gt-parent-dropout-v1-result.md.
Next permitted step is a small, separately frozen exact-GT augmentation training
smoke; full training/promotion is not authorized by data feasibility alone.
Antelume query againRequestExpired this turn; no RSNA/cloud mutations. Last
observed quota8.88h must be refreshed before any GPU launch.0/5qualified.

16:43 UTC: parent-dropout-training-v1/1 COMPLETE and HOST-VERIFIED. Final
NLL0.2183131454,parent2588/2645,absent11/27: unchanged gate FAILS (requires18).
All800 alternating fitting updates/nine actual checkpoints/runtime/controls
verified. Worker116.634s,launcher169.451s. No promotion/source/target expansion
or submission. Receipt854d4370...; see full parent-dropout result report.
Follower17836/download85259/verifier79273 terminal; no Biohub GPU job active.
Freshquota8.88h after completion,8h preserved. No RSNA/cloud mutation.0/5qualified.
This turn made progress through completed covariance and augmentation experiments,
not a mere status wait; neither produced a promoted candidate. Do not repeat the
failed head/threshold configuration unchanged. Goal remains unfinished.

16:38 UTC: full parent-dropout-training-v1/1 RUNNING, sole follower17836.
800 fixed alternating original/augmented fitting updates, original initializer,
same frozen encoder/flow, unchanged final real-data diagnostic gate. Actual
smoke-v3/1 passed and was host-verified:31.734worker/85.514launcher seconds,
4finite updates/exact checkpoint/full original controls,peak160MiB. No quality
claim from smoke. All19 local checks pass3.99s. Stage75571/push43098terminal.
Freshquota8.94h,0.25hcap preserves8.69h. Full result not yet available;0/5.

Verifier ready at scripts/verify-focus-parent-dropout-training.py. Recover all
runtime/result and nine checkpoint files after terminal completion to
.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1. Do not launch a
duplicate or tune against exposed diagnostics. Cloud/RSNA unchanged.

16:27 UTC: parent-dropout smoke-v2/1 accepted and running, a four-step fitting
functionality test only. Freshquota8.99h/0.25hcap preserves8.74h worst case.
Notebook2606a424...,490632bytes. Lossless packaging retry after v1 upload400
and confirmed404/no kernel. All15 local checks pass2.11s. No quality result yet.
The prior CPU audit verified1121eligiblepairs/1136synthetic absent-parent labels
without changing original or diagnostic packets. This addresses missing-parent
supervision, not a threshold sweep. Separate covariance screen failed and stopped.
See parent-dropout audit/design and correlated-motion result. No RSNA changes;
five qualified submissions remain0/5. No claim of current Antelume state.

16:12 UTC: joint training/1 COMPLETE and host-verified. All 800 updates and nine
checkpoints recovered; end-to-end 528.006 seconds. NLL 0.215789351, parent
2588/2645, absent 11/27: unchanged missing-parent gate FAILS (requires 18).
No source/target expansion or submission. Receipt f68b0762...; see joint result.
Follower 68068, download 78336 and verifier 95421 are terminal. No Biohub GPU
job remains active. Fresh Kaggle quota at 16:09: 8.99h remaining, only 0.99h
above reserve; recheck before any launch. Five qualified submissions remain 0/5.

Antelume named-profile query still returns RequestExpired; fallback default
profile has no credentials. Current instance state is UNKNOWN, not confirmed
idle or stopped. Local credential file last modified 14:18:52 UTC. No cloud
restart, shared environment or RSNA changes. Fresh Antelume credentials are
needed for further cloud resource inspection and GPU work.

Two newer public sources inspected with bounded findings, no execution or model
adoption; see public-refresh-20260910-1610.md. Neither establishes a qualified
candidate or superiority over the clean public best.

15:59UTC: full800step joint encoder/head training/1 RUNNING, solefollower68068.
Originalinitializer,12fit/4unchangeddiag, liveimageencoder gradients, tested
boundedAMPbackoff, detectorhead/flow frozen. Initialcompletecontrolreplay before
optimizer; unchanged finaldiagnostic gate. Periodicresumablecheckpoints.
Launched15:56 withfreshquota9.14h/0.5hcap preserves8.64h. Notebookd09c793a....
Fifteenregressiontests pass1.36s. Noqualityresult/submissionyet;0/5qualified.

Prior jointsmoke failed safely onnonfinitegradients beforestep2,70.604s;actual
failure verified. Backoffsmoke thenCOMPLETE82.027launcher/29.859worker seconds,
all4finitejointupdates,2safeskips,exactrealimagereplay/checkpoint,unchangeddetector/
flow. Peakallocated2.45GiB; hostreceipt1e904d0d.... Noqualitypromotion fromsmoke.
Allsmoke/download/verificationhandles terminal. AWSlastqueryRequestExpired;
no cloudrestart orRSNA mutation. CurrentGPUexperiment is Kaggle only.

15:42UTC: joint encoder/head smoke/1 launched, no quality result yet. Four
real-image fitting stress pairs, exact cached-feature replay beforeAMPupdates,
gradient/checkpoint/memory gate only. Original owned initializer; external
FOCUS detector and embeddedflow unchanged. Freshquota9.19h/1hcap preserves8.19h.
Notebook87f68814...,sole logfollower17930live. AWSqueryRequestExpired; last
successfulcloudstateSTOPPED, no restart orRSNA mutation.0/5qualified submissions.

15:33UTC: conditional motion produces a stronger full-source component, not a
promoted candidate.14movie-held-out correction screen PASS:10/14NLLmoviegains,
pooledMSE8.81605->8.51668,properNLL5.80412->5.74179,worstMSE20.1540->18.6545.
All14folds/final portable model exactly recomputed from saved training arrays.
Full-sourceCPU131.438s:combined0.7838611993,rawJ0.8012639505,divTP5/FP178/FN6,
vs previouscalibrated0.7805420622/rawJ0.7987813135/TP3. Source67ebd073 parent
delta-0.043314438 still violates fixed individual-movie loss bound. GateFAIL;
no target expansion/submission. Result998e43b1...,model508d0589.... NoGPUspent.
Sixteen focused tests pass4.22s; CPU63248/65605/test10166terminal.0/5qualified.
See conditional-motion training/source result reports for full scope/caveats.

15:20UTC: CPU fourteen-movie motion expansion COMPLETE118.172s, noGPU.
Original2movie Gaussian replay exact, then11,661ordinary links from14fit movies.
All8 full-source predictions saved beforeGT; patched controls/nodes replay.
Candidate combined0.7804171043,rawJ0.7993806382,divisionTP2/FP172. Previousfit
combined0.7805420622/rawJ0.7987813135/TP3. New fit FAILS true-division preservation,
previous-score gain and original per-movie maximum loss (67ebd073:-0.036827074).
No promotion/submission/target expansion. ResultSHA520a1e10...,fit379044ec....
Nine tests passed; CPU88667terminal. No live Biohub job or newGPU spend.
Five-good-submission request remains0/5. See expanded-motion result/design.

Host verification12619 subsequently COMPLETE: actual runtime, upstream feature
evidence, saved checkpoint hashes, original controls and recomputed unchanged
gate all agree. Worker resultSHA8707c59d9e7232afb788f45c9f364956d2ffe9526bcc0d297f2a6b4c180fbe08.
No active Biohub handles remain; failure is verified, not merely a log impression.

15:13UTC: null-balanced-head-v1/1 COMPLETE. Four-step real smoke passed, then
800steps finished115.907worker/169.913launcher seconds. Final unweighted
NLL0.216273710,parent2588/2645,absent11/27: missing-parent gate FAIL (needs18).
No extension or submission. Both checkpoints/runtime/logs downloaded; initial
control replay exact, frozen trunk unchanged. Follower65577/download49611
terminal; host verifier12619 pending at this timestamp. Fresh shared quota9.19h.
Antelume remains last observedSTOPPED; no cloud restart or RSNA mutation.
No Biohub GPU job active; five requested good submissions still0/5.

Public source-only review now includes sjlee1257, notebook998b7bc9.... No code
executed/weights adopted. It explicitly declares leaderboard-informed settings;
older packaged scorer and self-reported no-hack status are not certification.
See public-refresh-20260910-1510.md. Frozen-encoder head/calibration attempts
have not passed missing-parent feasibility; do not keep retuning diagnostic
thresholds or extend failed configurations unchanged.

15:08UTC: null-balanced-head-v1/1 launched after eight local tests passed and
fresh shared quota9.24h; declared1h cap preserves8.24h. Notebook SHA
2253c9927b1ccc6ee87f75e835c44665a164ee8aec78a12bcdb1a5994b59664e.
Twelve fitting movies10754parent/161null, unchanged diagnostic2645/27;
weight8.1728227 derived only from fitting counts. Exact original checkpoint,
forward/evaluation/checkpoint code retained; original control replay must pass
before optimizer, then four-step real smoke before800steps. No score yet.
Sole log follower65577. Antelume fresh EC2 check remainsSTOPPED, not restarted.
RSNA untouched. Five-good-submission request remains0/5; none qualified yet.

14:53UTC: both twelve-fit presence variants FAIL unchanged missing-parent gate.
Linear NLL0.186543,parent2602,absent7;fixed trees0.185738,2603,6; required absent18.
No source extension or submission. CPU fit resultsad1a546a.../4d0e314a....
Summary/1 COMPLETE135.844s launcher/85.491s worker,old-four exact replay; all
runtime/label/summary evidence host-verified. Handles6901/82699/67435/57230 terminal.

Prepared null-balanced head objective with fitting-derived weight8.1728227.
Seven local count/builder tests passed, then CPU-only Kaggle smoke/1 COMPLETE:
12.010s launcher/11.048s worker,original loss exact atweight1,unknown gradients
zero,correct null gradient/daughter labels,synthetic loss4.314->0.351. Actual
source/result host-verified; receipt1cd97274...,noCUDA/no real training. Handles
17538/45116/18151 terminal. No Biohub GPU job or weighted-head training queued.

Antelume CPU-test SSH attempt failed before connection; EC2 now authoritatively
STOPPED/publicIPnull (14:49 check). No instance restart, RSNA mutation or cloud
test occurred in that attempt. Latest shared Kaggle quota9.45h at14:45, with
recent RSNA notebooks on the same account; recheck before every launch.
Public metadata refreshed14:53, no new notebook execution/weights/selection.
Five-good-submission request remains0/5, no newly promoted candidate.

14:34UTC: expanded-summary/1 launched on Kaggle after prior featureCOMPLETE
and freshquota9.67h;1h cap preserves8.67h. Notebook5a67ae02...,sole follower6901.
Eleven prelaunch tests pass5.83s. Exact old-four fitting replay precedes new8
summaries; no optimizer/diagnostic evaluation in this GPU worker. No actual
expanded fit or new score yet. Antelume now has active RSNA PID2015, working
directory/data/rsna_round2_20260910,about8GB GPU allocation; left untouched.

Antelume small probe COMPLETE10.767s; original weights unchanged, four matrices
deterministic, zero real-parent argmax changes. Cross-Kaggle max error2.861e-6,
so bitwise replay FAIL; host verified actual matrices, no tolerance relaxation.
35.9MB peak model allocation,20% memory cap,180s timeout. Probe is terminal;
no Biohub cloud job still running. Isolated20.28MB package plus extracted files
remain/home/ubuntu/biohub_focus_head_replay_20260910_v1,not inside RSNA.

Extra feature/1 COMPLETE222.579s launcher/~177.3s worker. Bundle149910240bytes,
838files,archiveSHA7a165c6a.... Host verified all796packets,raw nodes/labels,
unchanged model/flow and replay; receiptSHA3d40210f.... Recovery72513 and
verification23962 and follower3994 are terminal. Original checkpoint recovered
locally,exactSHA76f7da6e...,19189835bytes. Five submissions today remains0/5;
no new model has passed complete-source promotion or submission acceptance.

14:23UTC: Antelume credentials/STS and SSH now work. Instance running at
3.236.42.34,oneA10G23028MiB,no compute processes in14:21 snapshot. Disk only
7.1GB free; RSNA data/shared environments untouched. User requested five good
submissions today; today0 submitted/0 new promoted. Fresh history latestAug26.
Subsequent cloud GPU work requires idle recheck, small replay, bounded isolated
storage and compatibility handling (sharedTorch2.5.1 versus Kaggle environment).

Additional detector cache/1 COMPLETE2627.502s, host verified800frames/324411nodes,
zero failures, exact six-frame replay. CPU label audit41.562s adds7936parent/
109absent; twelve-fit totals10754/161,original diagnostic2645/27 unchanged.
Raw receiptSHA55e3e18d...,labelSHAdba56011.... Feature/1 now RUNNING, sole
follower3994; notebook40115a14...,fresh launch quota9.74h/1h cap leaves8.74h.
Both small neural/flow replays passed; five new movies reported cached so far.
No optimizer, source/target evaluation, promoted candidate or submission yet.
Old follower99750 and download65777 and label process60311 are terminal.

14:09UTC: same live cache/follower99750 has reported seven full movies, latest
6bba_4f99ce20/6211nodes with zero failures. Last API status RUNNING; final movie
and terminal still pending. No partial cache promoted to label input. Prepared
label-free presence inference adapter: unchanged conditional parent ranking,
strict posterior>0.5 and max2children/max1parent. Forty focused tests pass5.14s,
including zero-correction probability replay and exact context equivalence to
the original summary calculation. This is not a fitted/validated candidate;
no new GPU job, fit, graph score or submission. Existing job remains intact.

14:00UTC: same follower99750 has now reported six complete100-frame outputs,
all zero failures (new16686/8778 nodes). Eight-movie terminal/verification still
pending; no labels from the new fitting movies opened. Prepared expanded
summary host verification and CPU fitting entry point. Thirty-three focused
tests pass6.74s, including a real original fitting-summary replay492parent/
20absent and corruption checks. Persistence-before-diagnostic and no-overwrite
tests pass; final eight-test rerun0.80s. No actual expanded fit, new GPU launch
or submission. The original fixed linear method SHA remains9c6c473a....

13:55UTC: API confirmed existing extra-fitting cache RUNNING; same follower99750
now reports four complete100-frame outputs, all zero failures (latest80491
nodes). Full eight-movie terminal/artifact verification remains pending.
Prepared expanded-summary builder for exact old-four fitting array replay,
then new-eight fitting summaries using the unchanged original head forward.
Original diagnostics neither recomputed nor evaluated on GPU; no optimizer.
Twenty-one related tests pass1.08s after fixing a source-replacement bug caught
locally. Neither feature nor summary follow-up staged/launched; no new-label
audit, actual expanded fit or submission. Previous turn classified progress
for completed tested feature-recovery/verification implementation, not merely
its live-status observation. Objective remains unfinished.

13:50UTC: the same Kaggle cache run remains active. Logs have reported three
complete100-frame inference outputs (6549,42172,53559 nodes), all zero failed
frames, plus both successful small replays. Full eight-movie terminal/artifact
verification is still pending; these log counts do not establish full completion.
AWS credential-file timestamp remains02:32UTC; Antelume utilization UNKNOWN.

Prepared the additional-fitting feature builder, isolated scope adapter,
bounded safe ZIP recovery and host per-file/model/label verifier. Twenty-two
focused tests pass, including isolated runtime imports and rejection of changed
staged identities, reordered frames, altered labels and sampler mutations.
Exact successful collector/math remain unchanged; original four diagnostic
movies are not extracted in the proposed follow-up. No staging, feature launch,
new-label access, fit or submission yet. Current detector is the only GPU job;
its1h cap and launch reserve accounting remain unchanged. No readiness claim.

Extra-fitting cache GPU logs now confirm models loaded on cuda0/cuda1 and both
three-frame replay checkpoints written with the expected hashes (785/285 nodes,
zero failed frames). Eight-new-movie completion is still pending. Continue the
same log follower99750; do not launch a duplicate cache or infer full completion.

Current GPU job: `biohub-focus-extra-fit-cache-v1/1` running on the next eight
fixed original fitting movies plus six replay frames. Original diagnostic4
unchanged; no new labels examined. Exactversion1 private/offlineGPU/noTPU
metadata and input list confirmed. Fresh launch quota10.48h,1h cap leaves9.48h
worst-case; expected35-50min. WrapperSHAbed7e37b...,sole log follower99750.
Thirteen focused checks pass; raw verifier/label inventory prepared but not
executed. No training follow-up queued or submission. Prospective twelve-fit
linear-presence protocol keeps method and original acceptance gates unchanged.

Latest turn was CPU-only. Fitting feature-separation audit21.25s/2818 pairs:
raw cosine79.67%, centered73.53%, L2 76.15%, physical91.66%. High average cosine
does not establish collapse; centering not adopted. New fixed shallow boosted
presence model completed8.406s; diagnostic NLL0.194705,parent2590/2645,
absent8/27. Same missing-parent gate FAIL (required18). No source extension,
target access, promoted model or submission. Native/export/JSON checks pass.
No Biohub GPU jobs launched or queued this turn; last quota10.48h. AWS
credential-file timestamp remains02:32UTC; utilization still not verified.

Latest completed result: parent-presence summary/1 COMPLETE112.933s launcher,
32.442s worker; verified runtime/model/labels and baseline replay. CPU-only
fixed logistic fit5.078s improves diagnostic NLL0.219009->0.193443 and correct
parents2585->2589, but known-absent correctness remains12/27 versus physical18.
Unchanged feasibility FAIL; no source extension, target access or submission.
ResultSHA89a6f4fe...,fitSHAf0752ddf.... Five tests pass. Fresh quota10.48h.
All Biohub GPU jobs now complete, no follow-up queued. Strongest submission
objective remains unfinished; source-best0.780542 is still unpromoted.

Latest public lineage source/12:31 revision screened at13:11UTC, SHAaeb5b94d....
It uses350ep public weights, feature TTA, harmonic fusion and a post-process
sweep; its receipt explicitly records leaderboard-informed configuration.
No settings, outputs or weights adopted. Not certified as current clean best.

New GPU summary-only job `biohub-focus-presence-summary-v1/1` running. Original
owned head frozen; new offset-logistic missing-parent intervention will be fit
on CPU after artifact verification. Five local tests pass; unchanged diagnostic
gates. WrapperSHAd298b109...,freshquota10.52h,1h cap preserves9.52h. Exactversion1
private/offlineGPU/noTPU confirmed. Sole follower56628. No source/target access,
actual new fit or submission yet. This supersedes older no-active-job snapshots.

Final artifact recovery/verification complete for head adaptation/1, including
both checkpoint hashes and actual runtime. Host recomputed the unchanged FAIL
gate. Launcher136.219s; worker93.532s, worker resultSHA8d8407da.... Fresh Kaggle
quota10.52h. No GPU run or follow-up queued; no new submission. Next hypothesis
is missing-parent discrimination using reusable features, not an active run.

Superseding latest status: head adaptation/1 COMPLETE800 updates, worker93.532s.
Final diagnostic NLL0.217096 (initial0.219009, physical0.333915), correct parents
2588 (2585/2500), correct missing parents11 (12/18). Frozen feasibility FAILS
the missing-parent count requirement. No source extension or submission.
All Biohub GPU jobs are now finished; no follow-up is queued. AWS remains
unverified due RequestExpired and no other workload was touched. Actual
checkpoint/runtime recovery is underway. Best source-development candidate
remains0.780542 and unpromoted, not comparable to a verified public best.

Head adaptation/1 confirmed RUNNING, private/offline GPU/no TPU exactversion1.
Four-step real smoke passed: nonzero finite gradients, frozen trunk unchanged,
strict checkpoint reload exact (step4 SHAc1608824...). At least500/800 updates
observed in existing log follower19949. Initial diagnostic: physical NLL0.333915,
2500/2645 correct parents and18/27 correct absent; initial neural NLL0.219009,
2585 parents and12 absent. Final feasibility unknown; these are original-training
diagnostics, not independent tracking scores. No new submission.
Correction: feature worker final elapsed156.861s;156.807s was its last-movie
progress time, not the terminal time. Feature launcher211.299s unchanged.

Feature cache/1 now COMPLETE211.299s launcher/156.807s worker. Host verified
834 required files and all796 feature packets (792 new/four replay). Exact
raw IDs, coordinates, sparse labels, frozen tensors, runtime, image and neural
replay all passed. No new tracking score. Twenty-seven focused tests pass.
Prepared and launching `biohub-focus-adaptation-head-v1/1`:800 transformer-only
updates, four-step real gradient/checkpoint smoke first, no diagnostic optimizer
access. WrapperSHAe2df68e7..., freshquota10.56h,1h declared cap leaves9.56h
worst-case. No automatic submission; full-source promotion still required.

Superseding update at12:46UTC: detector adaptation cache/1 COMPLETE35.51min,
800 new frames/174,818 nodes, zero failed frames; exact prior six-frame replay.
Host artifact verification and whole-movie sparse-label audit completed.
Fitting2818 parent/52 known-absent; diagnostic2645 parent/27 absent.
No optimizer has used these data. Diagnostic is not embryo-independent.

New single GPU job `biohub-focus-adaptation-features-v1/1` is running, using
the frozen owned encoder and flow on unchanged FOCUS proposals. Both small
GPU replays passed before new movies, including exact neural logits and
feature-save/reload replay. Fourteen local checks pass. WrapperSHA502bcc24....
Fresh launch quota10.62h,1h maximum preserves9.62h worst-case. Private,
two T4s, offline, no TPU; exactversion1 metadata confirmed. Existing log
follower96195 only; no duplicate job. AWS still RequestExpired, utilization
unknown, other projects untouched. No new submission or promoted candidate.

Existing GPU stream now reports four complete100-frame inferences with zero
failed frames (25,518/39,007/15,090/62,512 raw nodes). The full eight-movie
terminal is still pending. Prepared feature-pair identity/serialization checks;
11 focused tests pass. No real feature-cache or fine-tuning job launched.
Latest public combined-knobs source supplies no verified new trained model;
leaderboard-selected settings were not adopted. Best candidate remains
unpromoted, and no submission has been made.

Training loss wiring now passes a synthetic CPU test with verified runtime,
40 optimizer steps and exact checkpoint reload. This consumes no GPU and
does not establish biological quality. See `focus-indexed-loss-cpu-v1-result.md`.
The sole GPU cache run continues; its existing log stream reports two complete
100-frame inferences (25,518 and39,007 nodes), zero failed frames. Eight-movie
completion has not been observed. No fine-tuning or second GPU job is queued.

Adaptation cache/1 remains RUNNING; logs at12:13UTC confirm both six-frame
replay artifacts with expected hashes and zero failed frames. No completed
eight-movie result yet. Prepared label audit now passes11 small tests and an
exact176-transition existing-training replay. A discovered edgeless-scorer
shortcut was handled through the same underlying DistanceMatching, without
artificial edges. No adaptation training starts until the cache and label
coverage are verified. No second GPU job or new submission was launched.

Current GPU work: `indarkarhana/biohub-focus-adaptation-cache-v1/1` is RUNNING.
It caches raw FOCUS detections on eight original training movies: four for
potential adaptation and four for its diagnostic, plus six replay frames.
No source-selection or target data, no optimizer and no submission. Previous
matching cache took34.8min; estimated35-50min, hard cap1h. Fresh launch quota
11.22h preserves10.22h worst-case, above8h reserve. Frozen notebookSHAefd1cdef....
No other Biohub GPU job overlaps; no training follow-up is automatic.

The completed two-movie label inventory exposed inadequate missing-parent
coverage: fitting708 positive/3 known-absent, diagnostic112 positive/0 absent.
Unknown cells remain ignored. Nine label/split tests pass. Also corrected the
previous explanation: original training already uses native predicted nodes;
FOCUS-specific proposal-domain adaptation, not predicted-node training itself,
is the untested intervention. See `focus-predicted-node-labels-v1-result.md`
and `focus-adaptation-cache-v1-design.md`. Best score remains0.780542 and is
not promoted. This cache job does not establish a submission candidate.

Latest owned learned-linker follow-up COMPLETE62.929s. Raw FOCUS detections
plus our frozen neural head retains1,070 nodes but does not recover any correct
links in the six-frame training probe:19TP stays19, FP3->7, false divisions0->2.
Raw edge Jaccard0.73077->0.63333, so the frozen feasibility gate FAILS.
No full-source extension or submission. Nine focused tests and actual runtime,
tensor, node and graph replay checks passed. Latest Kaggle quota11.22h, above
the8h reserve. No Biohub jobs active or queued. See
`focus-owned-neural-probe-v1-result.md`; best development score remains0.780542
and unpromoted. Predicted-node training is an untested next hypothesis, not
an active training run or an established fix.

Latest completed follow-up: CPU-only registration anchor probe/1 finished
61.890s. Twelve focused tests passed; runtime and private/offline CPU metadata
verified. Six original training frames only. Its fixed correction fails the
21-link training feasibility screen: squared residual7.01758->7.02045 and
one movie worsens. Rejected without a full cache or GPU run. No source/target
access or submission from this probe. No Biohub jobs are active or queued.
See `focus-registration-probe-v1-result.md`. Best development score remains
0.7805420622 and still fails the unchanged promotion gate.

Superseding snapshot at 2026-09-10 11:40 UTC: no Biohub GPU or CPU experiment
is running and no follow-up is queued. AWS DescribeInstances again returns
RequestExpired; credential-file mtime remains 02:32:02 UTC. Antelume GPU
utilization is UNKNOWN, not confirmed idle. No other workload was changed.
Fresh Kaggle quota is 11.24h, leaving 3.24h above the required 8h reserve.

Training-only residual calibration completed on CPU in 75.704s. Fixed-eight
source score improved from 0.7753325939 to 0.7805420622; raw edge Jaccard
improved from 0.7937315832 to 0.7987813135. Mean node recall is unchanged;
false divisions fell from 186 to 154 with three true divisions retained.
However, movie 6bba_67ebd073 remains 0.0247114 below the original parent,
outside the frozen 0.02 limit. NOT PROMOTED. No new target movies or submission.
This is source development evidence, not independent or leaderboard evidence.
See `focus-residual-calibration-v1-result.md` for fit provenance and hashes.

Completed CPU attribution identifies 613 missed links with detected endpoints:
540 outside the fixed motion gate, 72 lost to parent competition, and one
topology-blocked. Another 271 misses lack detected endpoints. These diagnoses
do not establish that relaxing the gate or adding those links is valid.

Current GPU work: none. Real-image restoration v2 ended ERROR after100
updates (119.523s), with resumable checkpoints and full diagnostics recovered.
Its quality proxy FAILS: zero of24 diagnostic movies improve; MSE56.86x the
center-excluding neighbour mean. This checkpoint will not feed FOCUS.
Tiny GPU center derivatives also fail the strict-zero audit despite exact
zero finite-perturbation response; numerical explanation remains unconfirmed.
That issue cannot rescue the failed quality proxy. No GPU extension, detector
rerun, tracking-selection/target access or submission. Fresh launch quota
was11.28h with1h cap/reserve8h. See `blindspot-real-probe-v2-result.json` and
`blindspot-restoration-v1-design.md`. Stronger existing FOCUS-flow evidence is
unchanged, including its still-unresolved per-movie regression.

Latest CPU-only ablation completed: fixed-division joint assignment gives
0.7749123901 versus the FOCUS-flow reference0.7753325939, with11 more correct
but18 more false edges. It worsens67ebd073 and fails acceptance; rejected
without a GPU rerun.14 small tests passed, all predictions preceded GT, and
both controls replayed exactly. No new submission or GPU job. Details:
`focus-division-preserving-assignment-v1-result.md`.

Superseding execution status: source route COMPLETED10:33:11UTC. Both GPU
jobs and local CPU replay/scoring finished; no Biohub GPU follow-up queued.
FOCUS plus owned flow scored0.7753325939 versus parentD4 0.6298395328 and
same-FOCUS static0.7496408177 on the fixed8 complete source movies. Seven
movies improve versus parent; mean recall0.9643974 versus0.9575953. However,
6bba_67ebd073 adjusted-edge score regresses0.0381222, exceeding the frozen
0.02 limit. SOURCE GATE FAILED; retain the component evidence but do not
promote or open new target movies. No submission or public-best claim.

All168,586 source detections were preserved; exact motion/graph replay and
fresh parent re-score passed. Detector elapsed2090.451s; motion133.158s.
Fresh motion launch quota11.35h,1h cap,reserve8h. Result:
`focus-source-flow-v1-result.json`, SHA
`896d3de86fa30fcd860f2c16073d47718118d94e1a43fe70710626c7b32ea169`.
The complete-movie comparison is source development evidence, not independent
embryo or leaderboard evidence. FOCUS pretraining overlap remains unverified.
See `focus-source-route-v1-design.md`.

The CPU-only identity audit verified author nuclei weight SHA b14a7bd2...
and29 runtime entries. The subsequent GPU smoke COMPLETE437.508s and host
replay verified all1,070 coordinates exactly;21 related tests pass. A slow
recursive reference lookup identified in the smoke has been replaced in the
full-cache builder by tested explicit mount paths. FOCUS pretraining overlap
remains unverified, so identity success is not independent accuracy evidence.

Fresh user-requested execution check at approximately09:30UTC: Biohub's last
GPU notebook is COMPLETE; no Biohub GPU run or automatic follow-up is active.
Kaggle quota12.07h, leaving4.07h above the required8h reserve. AWS named-profile
DescribeInstances still returns RequestExpired; default profile has no
credentials. Read-only SSH to the last recorded Antelume address timed out.
Therefore Antelume utilization is UNKNOWN, not confirmed idle. No other
project's process, storage or instance state was changed.

New completed CPU experiment `public-owned-flow-repair-v1`: reuse our verified
native image-flow graph as continuation support for the stronger public
control, with the existing frozen3um mutual-match/no-reassignment policy.
13 small tests passed. All12 graphs were persisted before opening labels;
exact cached motion/graph replay and prior-consensus replay were verified.
Result: ZERO eligible added edges across all4 movies. Candidate equals public
control at0.9440079104, below prior consensus0.9447251850. Both required gain
comparisons fail. This route is rejected without a GPU rerun or submission.
These are base-training diagnostic scores, NOT leaderboard or independent
validation scores. All previous target restrictions remain unchanged.

Latest public date-run listing was refreshed successfully with UTF-8 output.
Newest entry remains the already-screened DAE self-distillation notebook;
no known metric-exploit notebook was pulled or run. No new public-best claim
was verified. Additional GPU access alone does not resolve the missing
competitive/generalization evidence. No same-day score guarantee is supported.
Result: `public-owned-flow-repair-v1-result.json`.

The completed motionD4 transfer diagnostic FAILED its frozen gate: aggregate
score+0.0017010553 and raw-edgeJ+0.0017217184, but only2/4 movies improved
(one unchanged, one regressed). Correct edges remain593; false edges156->153,
false divisions36->35. No threshold/gate relaxation or new target access.
Retain the original parentD4 reference for promotion comparisons; record the
motion source gain without treating it as transferred generalization.
Transfer CPU controller terminalcompleted.

The distinct cached-FOCUS/owned-flow GPU probe is COMPLETE and host-verified:
285 exact nodes,174 edges per arm,3 training frames,57.023s total launcher.
No accuracy gain is established by that smoke. The subsequent four-complete-
movie test is now COMPLETE:0.7857341462 static ->0.8103292365 owned flow,
all four movies improve and its diagnostic gate passes.47 more correct edges,
13 more false edges,11 more false divisions. The same65,091 raw nodes remain.
GPU launcher144.589s; full CPU replay/scoring completed09:16:58UTC. These
previously exposed movies do not establish independent generalization; the
historical public-neural raw linker remains higher at0.8802090945. No claim
of beating public bests or submission readiness. No active Biohub GPU job.
See `focus-owned-flow-full-v1-design.md`; no large detector rerun or submission.

Historical motion launch/result notes:

Motion-only source validation is now COMPLETE and passed its predeclared
gate: score0.6321840591 (+0.0023445263), raw-edgeJ+0.0021527269, unchanged
node recall and detections;5/8 movies improve.22 more correct edges,7 more
false edges. This does not establish public-best performance. The one-shot
source CPU controller completed. A frozen same-four-movie transfer diagnostic
is staged with73 relevant checks; its design and notebook hashes are in
`flow-tta-transfer-v1-design.md`. Transfer GPU version1 is now accepted and
RUNNING; fresh launch quota12.30h, declared1h. One-shot CPU scoring/comparison
controller PID40824 is live, started08:52:27UTC; logs are under
`.biohub/automation/flow-tta-transfer-v1/`.40 focused contract/comparator/queue
checks pass. No new target movie or submission yet.

Historical launch note (superseded by the completed source result above):

Motion-only full-source validation `biohub-flow-spatial-tta-selection-v1/1`
is now launched. It freezes parentD4 detector coordinates against the original
eight saved graphs, averages only flow fields, and uses the GPU-tested zero-
neural shortcut.43 inference/scorer/reference tests passed before launch.
Fresh quota12.59h, declared1h cap, reserve8h. One-shot CPU controller PID35396
will launch the short-name frozen scorer and write the comparison. No complete
motion score, target opening or submission yet. See flow-spatial-tta-selection-design.md.

Previous completed results:
The fixed detector ensemble is **REJECTED** after complete eight-movie official
scoring:0.5039128669 versus retained parentD4 baseline0.6298395328. Raw edge
Jaccard fell0.0614372; all8 movies regressed. No new target movies or submission.
GPU inference took1772.127s. Its51-character CPU title/slug caused an upload
failure; a shorter metadata-only repair was accepted and the exact frozen CPU
notebook completed. Report:owned-detector-ensemble-selection-v1-result.json.

The separate motion/runtime probe v2 also completed (71.295s,29 tests passed).
Skipping zero-weight neural execution preserved the real three-frame graph
exactly. MotionD4 preserved nodes/weights but added one false edge on that tiny
training sample; it is not yet a complete-movie accuracy gain. At that point no
GPU job remained; the next motion-only validation above is now active.

Earlier execution history:
Fixed ensemble smoke `biohub-owned-detector-ensemble-probe-v1/1` passed17 GPU
tests and the three-training-frame graph check in73.116s. This is functionality,
not accuracy evidence (tiny training edge counts worsened). Strict checkpoint
and frozen tensor receipts verified. Complete eight-source ensemble inference
`biohub-owned-detector-ensemble-selection-v1/1` has now been launched on Kaggle,
one-hour watchdog,13.12h available immediately before launch. Original cutoff,
equal probability weights, per-model detector D4, parent native features and
standalone learned flow unchanged.42 local inference/scorer regression tests
pass. CPU scoring staging is frozen and will follow completion. No new target
movies or submission authorized. No claim of beating public bests.

Previous completed branch:
Current status: `biohub-owned-detector-calibration-full-v1/1` is COMPLETE
(854.645s launcher). All360 artifacts and18 exact probe replays verified.
Full calibration FAILED: diagnostic569/597 matches vs parent574/597; four
movies regress, including7 lost parent matches on6feb10f0. The cutoff
0.993919312953949 is rejected. The source-validation builder correctly refuses
this real failed report, and no calibrated notebook was created/launched.
Before the new ensemble probe/validation there were no active Biohub jobs;
quota was13.14h. No calibrated cutoff is approved.
Both the PU transfer diagnostic and six-training-movie calibration probe are
COMPLETE. Fresh full-collector prelaunchquota13.38h;1h cap preserves12.38h,
above the required8h reserve. No competition submission.

PU transfer FAILED: aggregate0.5787726181 vs baseline0.5698400154, but raw
edge Jaccard-.0022799011, two movie regressions, worst-movie loss.0434388987.
Both models recovered593 annotated edges; PU adds4 false edges. Do not extend
this exact PU checkpoint/policy to the remaining65 target movies or submit it.
The measured source gain remains recorded, not retroactively erased.
Report: `owned-detector-pu-transfer-v1-result.json`. Zero new target movies opened.

The next distinct branch tests training-only confidence drift of the sparse
detector. Small collection `biohub-owned-detector-calibration-probe-v1/1`
completed94.121s launcher /37.521s collection. All18 frame artifacts, original
training annotations, checkpoint and source hashes verified; weights unchanged.
A probe-only cutoff0.9942006469 preserves140/149 fitting matches, equal to the
parent; exact rematching preserves46/46 diagnostic matches across two movies.
One fitting movie loses one annotation despite the pooled fitting constraint.
This passes the small functionality/diagnostic check, not full calibration or
independent validation. Full96-fit/24-diagnostic training collection has now
launched after13 prelaunch tests and requires18 small-probe replays. Do not
deploy the probe cutoff or tune it on source
selection/target scores. Report: `detector-calibration-probe-v1-result.json`.

Full outcome supersedes the successful small-probe check:96-movie fitting
retained2582/2707 vs parent2595/2707; diagnostic baseline sparse recall589/597
fell to569/597 after calibration, versus parent574/597. The global confidence
cutoff failed to transfer across the training diagnostic movies. No threshold
sweep or small-probe-cutoff substitution will be performed. Full report:
`detector-calibration-full-v1-result.json`. All18 replay probability differences
were exactly0, so this is not a numeric/runtime mismatch.

The two-T4 paired detector fit `biohub-owned-detector-fit-pair-v1/1` is COMPLETE
and verified. Each arm has1,000 updates; sparse control then raw-logit view-consensus
positive/unlabeled detector training. Same initialization and recorded data;
teacher, linker, flow and BatchNorm statistics fixed. The sparse arm has reached
1,000 updates (773.482 optimization seconds); PU completed in638.578s.
Total launcher1644.197s. Training and strict reload passed, not submission validation. Prelaunch quota
14.67h; one-hour total cap. AWS named-profile read-only access still returns
RequestExpired; default has no credentials. Antelume utilization is unknown.
No other project's workload was changed.

All ten-update numerical/functionality probes are complete. Raw-logit target
construction fixes sigmoid saturation without count-based threshold tuning.
The paired training receipt is verified. The sparse arm's eight-movie score is
0.2491592563: rejected against frozen-D4 0.6298395328. Raw edge Jaccard also
fell0.0113285134 despite recall rising0.0179729683.910864 detections versus
219373; all8 adjusted-edge scores regress. PU GPU inference and CPU scoring
are now COMPLETE. The queue finished successfully at06:25:37UTC with no
submission or new target audit. Biohub has no active GPU experiment
at this point. This does not establish that the shared AWS GPU is idle:
a fresh read-only Antelume query again returned RequestExpired.
41 local inference/provenance tests pass. Compare
both arms with the frozen D4 baseline and PU with sparse control; retain
per-movie/recall safeguards. Remaining65 target movies are still closed.

**Measured progress, not submission readiness:** PU scored0.6506121996 versus
frozen-D4 0.6298395328 (+0.0207726668). Raw edge Jaccard increased0.0041955567;
mean node recall changed-0.0003210486. Six of eight adjusted-edge movie scores
improved; two regressed by less than0.006. Worst-movie adjusted-edge score
improved0.0099110451. Counts:5451TP1159FP1359FN, versus5440TP1192FP1370FN.
This passes the predeclared comparison with the strongest frozen baseline.

However, the additional PU-versus-sparse-control recall condition fails:
recall-0.0182940169, exceeding the0.005 allowance. The registered combined
gate therefore chooses no arm. Do not change this rule after seeing results
or misread the report's overall rejection as absence of baseline improvement.
The exact comparison is `owned-detector-selection-v1-comparison.json`.
The frozen PU checkpoint remains a research lead, not an authorized submission
or a demonstrated public-best replacement. Its target-embryo transfer has not
been evaluated. A separately declared follow-up must preserve this failed
gate and distinguish any already-exposed diagnostic movies from new holdouts.

Latest quota after both GPU jobs completed:13.57h remaining;5.57h spendable
above the eight-hour reserve, subject to a fresh check before another launch.

The bounded local evaluation queue started at2026-09-10T05:33:52Z, PID40244.
Logs and terminal state: `.biohub/automation/owned-detector-evaluation-v1/`.
It waits for verified paired training, launches the two source-selection arms
sequentially with fresh quota checks, CPU-scores them, and writes
`owned-detector-selection-v1-comparison.json`. Maximum queue duration3hours;
any failure stops progression without retries or changes to remote jobs.
The sparse source-selection version1 was accepted at05:42:30Z after freshquota
14.21h; completed1227.423s with all800frames. Sparse CPU scoring version1
accepted06:04:24Z. PU selection version1 accepted06:04:33Z after freshquota
13.86h; one-hour cap preserves12.86h. No submission or target audit is scheduled. Do not manually duplicate the
queued selection/scoring launches.37 relevant tests plus the expanded8 queue
tests pass. The workstation must remain running for the local queue to progress.

While GPU inference ran, two CPU global-assignment experiments completed.
Static-motion assignment scored0.609385368; cached learned-flow assignment
scored0.623818462. Both remain below frozen detector-D4/flow0.629839533;
neither is promoted. Frozen-source error attribution found999 missed annotated
edges whose endpoints were detected versus371 with missing endpoints. This
supports investigating association quality, not assuming a larger detector
alone will solve the tracking errors.

A training-only recall-constrained confidence-calibration primitive and test
design are prepared, not a completed calibration experiment. Ten unit tests
pass; the real GPU collector is not implemented or scheduled. Together with
submission-shard duplicate detection and error-attribution tests,27 tests pass.

## Independent development evidence

These are patched-official scores on the same eight complete source-embryo
selection movies excluded from training, not leaderboard scores. Fixed-detector
linker comparisons preserve 221,567 native detector nodes; detector D4 is an
explicit changed-detector experiment with219,373 image-derived nodes. The first
four of69 target-embryo movies completed the frozen D4 audit at0.5698400154;
the other65 remain unevaluated for this family.

| Model/change | Selection score | Decision |
| --- | ---: | --- |
| Native joint detector/linker, c502 | 0.602966 | Reference |
| Longer joint fit | 0.535490 | Reject: six movie regressions |
| Detector BatchNorm recalibration | 0.569477 | Reject: six movie regressions |
| Linker feature-only spatial TTA | 0.602418 | Reject: six movie regressions |
| Known-missing-parent null training, b642 | 0.608901 | Best trained neural-linker component |
| Division-specialist fine-tuning | 0.604976 | Reject: more false divisions, below b642 |
| Static spatial association prior | 0.600549 | Motion control |
| Causal motion history | 0.619712 | Previous strongest standalone |
| Learned backward image motion, 3006 | 0.623256 | Previous strongest source-selection reference |
| Frozen image motion + trained b642 linker, 76f7 | 0.610494 | Below standalone flow: do not promote |
| Training-only calibrated image motion + neural scores | 0.603207 | Reject: below uncalibrated model and standalone flow |
| Detector-only D4 averaging + frozen standalone flow | 0.629840 | Source gain, but target-first4 score0.569840 is not competitive |

The backward-flow gain is +0.003544 over causal motion, but four of eight
movies regress against that control. It gains 108 true edges and also adds
126 false edges. The worst movie is `6bba_c73a1d11`. This is useful development
progress, not sufficient evidence to ship a public-best-beating model.

The integrated 76f7 fit completed 1,000 updates in 660.60 seconds including
bootstrap/checks. Both detector and flow remained frozen; all 200 first-probe
input hashes replayed. It preserves the trained neural linker and replaces
only the spatial prior before further linker optimization. Selection evaluates
all eight complete movies against b642, c502, causal motion and standalone
flow. Inference completed in 267.53 seconds with all frames and exact native
detector nodes preserved. CPU scoring caught an inherited manifest-name mismatch;
the repair verifies that name against the original hash-checked runtime without
modifying prediction artifacts. CPU scoring version 2 completed; no GPU rerun
was needed. The integrated model recovered 310 more true links than standalone
flow but introduced 634 more false links. Six movies regressed against flow;
four regressed against its b642 parent. It is not the strongest candidate.

The next experiment completed a small 18-pair functionality probe followed by
360 training-only pairs. The frozen-network calibration fitted three bounded
coefficients: neural weight 0.398903, spatial weight 0.309181, null score
-2.832382. All probe input/cache hashes replayed. Fitting used 2,144 annotated
columns including 24 verified missing parents. On 503 calibration-diagnostic
columns, confident wrong decisions fell from 81 (flow) to 53, while correct
decisions rose from 422 to 445. These movies are within the neural model's
training pool: this is **not independent full-movie evidence**.

The calibrated inference wrapper passed 10 CPU tests. Full eight-movie inference,
CPU official scoring and strongest-control comparison are prepared. A shared
GPU slot freed when RSNA math-light completed; calibrated validation version 1
was accepted without interrupting RSNA paired-TTA. Inference and CPU scoring
completed: calibration lost 0.007286 against the unchanged neural model and
0.020048 against standalone flow. All eight movies regressed against flow.
It added 110 false edges for one additional true edge versus the uncalibrated
model. The calibration-diagnostic benefit did not transfer; do not promote it.

The next distinct branch averages eight inverse-aligned detector views while
keeping the motion model on the original image pair and fixing the linking
policy. Its three-training-frame probe passed: 292 to 269 detections, 12 to 14
true links, 6 to 5 false links. These tiny training results are functionality
evidence only. Full eight-movie detector-TTA validation version 1 is running.
The predeclared comparison requires raw edge Jaccard and combined-score gains,
with no more than 0.005 absolute mean node-recall loss; node-count reduction
alone is insufficient. Per-movie failures remain visible.

## Public-source evidence and exclusions

An earlier public-base diagnostic scored 0.944008 on four movies; our FOCUS
consensus diagnostic reached 0.944725. These are **not independent validation**:
the secondary checkpoint trained on all 199 movies, and exact original
training membership for the support50 primary remains unverified. Do not
compare these numbers directly with the independent eight-movie scores above.

Latest public source screening did not establish a new independently validated
checkpoint. See `public-refresh-20260910.md` for URLs, hashes and decisions.
Known older fabricated-node/time metric exploits were skipped. No new public
notebook was executed as a replica, and no graph-count manipulation is allowed.

## Remaining gates

1. Detector-only D4 validation and CPU official scoring completed. Score
   0.6298395328 (+0.0065838962), raw edge Jaccard +0.0062465435, mean node
   recall -0.0000354034.19 more true links,46 fewer false links; six of eight
   movies improve. The predeclared source tracking/recall gate passes.
2. Frozen first-four target-embryo audit completed with unchanged model/policy:
   score0.5698400154, mean node recall0.8583714789,593TP156FP269FN. This does
   not support a competitive submission.65 target movies remain closed; do
   not convert these four into a threshold-tuning pool. The preplanned bounded
   image-loss flow component probe completed without a paired gain and will
   not be extended. A detector-only positive/unlabeled training probe passed
   execution checks but regressed on three training frames and exposed FP16
   teacher-probability plateaus. A separate FP32-corrected ten-update probe is
   now launched after11 CPU tests. This is a functionality test, not a claimed
   competitive candidate. No full-test inference launch is justified yet.
3. Pass independent embryo reporting, worst-movie analysis, offline inference,
   full-test runtime and CSV/graph integrity checks before submission.

## Resources

Antelume's last successful read-only check reported STOPPED. The latest query
failed RequestExpired, so its current state is unknown. No AWS instance or
RSNA workload was changed. Kaggle ran the completed fit and
then the integrated selection sequentially, offline on the two-T4 machine.
Quota immediately before selection was 16.91 hours; its one-hour worst-case cap
preserves 15.91 hours, above the required eight-hour reserve. Latest quota after
completion of the integrated experiment: 16.64 hours. Latest prelaunch check
after calibration: 16.29 hours. On continuation, quota was refreshed to 16.18
hours and calibrated selection version 1 launched with a one-hour cap. It and
CPU scoring are complete. After the detector-TTA probe, fresh quota was 15.62
hours; detector-TTA selection launched with a one-hour cap, preserving at least
14.62 hours worst-case. Detector-TTA completed in938.326 seconds. Fresh quota
before the target-embryo audit was14.98 hours; its one-hour cap preserves13.98
hours worst-case. Actual accepted audit slug is
`indarkarhana/biohub-detector-spatial-tta-embryo-audit-v1/1`. The GPU notebook
is immutable; CPU scorer input uses this actual slug. No competition submission
has been made in this iteration.

Latest fresh quota before the corrected detector probe:14.72h;1h cap preserves
13.72h worst-case. Active accepted kernel:
`indarkarhana/biohub-owned-detector-pu-fp32-probe-v1/1`. The image-motion probe
and original PU detector probe are terminal; no Biohub GPU experiments overlap.

Authoritative receipts live in `reports/experiments/*-score.json` and
`image-motion-linker-v2-training.json`; this document is a human-readable snapshot.
