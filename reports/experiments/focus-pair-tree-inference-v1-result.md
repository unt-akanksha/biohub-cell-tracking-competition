# Native inference and independent verification preparation

September10,20:56UTC. The12fold nonlinear quality run21863 remains live; five
folds complete, sixth fitting. No aggregate acceptance or submission candidate
yet. This turn advanced inference correctness/runtime and tested the verifier
without changing any launched model, feature, threshold, objective or gate.

## Actual complete-frame timing

The original frame has1160source and1187target detections:1,378,107 complete
real/null choices, including1167originally unknown targets. Labels were removed
before prediction, and every source/target ID, coordinate and input array was
preserved. All native/portable decisions and selected/null posteriors match.

| Fixed model on this same frame | NativeCPU seconds | Portable seconds | Max posterior difference |
| --- | ---: | ---: | ---: |
| Original20-group smoke model |5.062|15.063|0|
| Full11-movie-trained first-fold model |5.907|13.109|0|

These are single measurements under concurrent local CPU load, not universal
speedups. Native prediction uses the exact100tree structure, pinnedXGBoost3.4.1,
2threads andCPU, adding the external full linear baseline once. Different
target block sizes32native/17portable independently check chunk invariance.
Empty frames, duplicate node identities, wrong baseline dimensions, complete
unknown coverage and bad block sizes are covered by tests.

Original smoke receipt272b2e796c853871927306aa99eea44a30ffc281e0dd19b7c4129ca5828eebcf.
Full-model runtime/workload receiptf9f94cd5c4283c794d8b481bb539b3433a37bf1fd960d5cf5b9cd1171fa089ea.
Native outputsd534e79ab101b0ca47cc7f0a0195bf46c80d27f88ac535f948ac55fe743b0c41.
Only the original preselected fitting packet and fixed first-fold model were
used; no runtime-model selection based on held-out quality.

## Workload accounting, not a hidden-test runtime claim

All12 original hash-verified fitting manifests cover1188transitions and462,075
target decisions,451,160originally unknown. Complete inference requires
302,977,426candidate/null choices,64.58times the4,691,320known-label training
choices. All original frame counts fit the existing2048node capacity. Maximum
conservative32target block estimate158,108,672B, not measured processRSS.

Linear work scaling from the single full-model frame suggests1298.66seconds
for this12movie candidate-scoring workload ONLY. This is not a measured movie
runtime, excludes image I/O/encoding/detection/flow/graph work, and does not
establish12hour hidden-submission feasibility. Do not budget inference using
only the labeled training rows or use this estimate as a submission approval.

## Verifier readiness and correct launch path

scripts/verify-focus-pair-tree-lomo.py is implemented and smoke-tested on the
actual first completed held-out fold. All231,230choices,417/492parent and11/20
absence decisions, NLL0.54899254935 independently replay via native predictions,
17group blocks and per-group logaddexp/argmax. Native/portable error0;3.031s.
Receipt29a30bfe37fab57eceb47065cb052c3e713c6941f81132dbabb4f525860ce163.
Verifier sourcecf2f8058406b251617aea3c28f43196e3e81ffcecc41d8ed9ebc1798f805a2ed.
This smoke did NOT verify all12folds or their full fitting objectives.

After all12finish, run from the pair-tree environment:

    .biohub/cache/pair-tree-venv/Scripts/python.exe -u scripts/run-focus-tree-verification.py

Use this new controller, not the verifier's legacy default subprocess entry.
It targets the real base interpreter directly, validates actualPID, enforces
1800s/4GiB, requires5GiBfree memory, and preserves its own receipts. This avoids
the already diagnosed Windows venv-launcher accounting/termination problem.
Full verification replays every original11movie prepared feature/margin/group,
fold-only projection/count/weight, native fitting loss, saved25tree checkpoint
prefix, held-out result, comparison control and unchanged five-part gate. It
does not fit an optimizer.30 related tests passed7.87s; full verification has
not started while the quality experiment is live.

## Public/account refresh and resources

At20:43-20:46, the20-topic inventory IDs/comment counts matched the19:23cache.
This is a metadata comparison, not a fresh reread of every discussion body.
The recent12notebook dateRun list shows no new revision beyond the previous
refresh. Already excluded dhiaalhemdani revision19:43 was not downloaded; no
public code/weights were pulled or adopted in this refresh.

At20:52, own submission list still begins with55784044 (Aug26 COMPLETE, blank
score) and55364415 (Aug9 COMPLETE, visible0.912). Blank scores are not inferred
from notebook titles/descriptions. No new submission from this work;0/5.
No GPU/cloud/RSNA mutation. Inference/profile/test sessions are terminal; only
quality session21863 remains live. Current rough remaining time35-50minutes
for this screen plus independent verification, not for a guaranteed candidate.
LastKaggle8.22h19:17 remains historical and must be refreshed beforeGPU use.
