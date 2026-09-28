# Exact event pruning acceleration, September14,2026

Research-only runtime work while the original fork16 fits continue unchanged.
The objective is to make dense full-movie evaluation/submission inference more
practical without changing event scores, biology, graph constraints or metrics.
No public notebook replica, node-count rule, selector or model retuning is added.

## Component proof complete

`trajectory_event_vectorized_dominance_v1.py` vectorizes the existing exact
fork-vs-continuation-plus-birth dominance test. It retains the same strict
1e-10relative numerical tie tolerance and respects allowed-option masks.
Integer key overflow falls back to the unchanged reference implementation.

Ten real frozen source cases were selected by uniform case-order coverage plus
largest option vocabulary within each source embryo, not by scores. Sixty
comparisons cover fixed source-prior anchors, zero and seeded random weights,
with all options or seeded random partial masks. No source target/safe/margin
arrays, intermediate learned checkpoints, selection or validation labels loaded.
All60retained masks and pruning-count dictionaries matched exactly.

Reference pruning16.4143345seconds versus vectorized0.5923135seconds across the
comparisons:27.712x for this component only, not an end-to-end speedup claim.
Full benchmark18.188seconds, session2762 terminal exit0. Result is
`trajectory-event-vectorized-dominance-v1.json`. Experimental helperSHA
`23db55460d7e7cd079d8b03e36c9219c64ab4e4c348c01ee30f1194b3d9da0f8`.
The original helperSHA remains
`e8f7daa6b05ece8803c75313b743ce3a3d8ae1816e6203d1d54f530e381a5c3e`.

## Full-movie proof now running

`trajectory_event_vectorized_inference_v1.py` uses private function namespaces
to replace only the pruning function. Original inference/solver function bytecode
and all keyword defaults are retained; imported reference modules are not mutated.
Fifteen targeted tests passed in5.25seconds, including100random partial-mask
problems, shuffled options, numerical near ties, forbidden births, exact solver
choices, zero-gain/fork graph identity and reference namespace isolation.

`replay-trajectory-event-vectorized-inference-v1.py --held-embryo 44b6` is running
as session7794. It replays the two already-frozen control movies with the SAME
10second/frame600second/stage local budget. It must reproduce complete graph
JSON hashes and every solver/edit record except elapsed seconds exactly, with
no fallback or budget exhaustion. No GT/scoring/model fit is involved.
The reciprocal held6control is now also complete and available for a subsequent
replay. No newly vectorized code is deployed or installed in either live fit.
Full learned-model and Kaggle runtime acceptance remain unestablished.

## First complete replay passed

Session7794 is TERMINAL success,177.735seconds total. Both100-frame control
graphs reproduce exact JSON hashes, all99transitions and every solver/edit
record apart from elapsed time. No fallback/budget exhaustion, GT/model change.

-44b6_996155de:10.453s reference versus10.719s vectorized; no speedup here.
-44b6_d5e7d891:210.578s reference versus162.375s vectorized,22.89%less elapsed.

These are single local runs during the unchanged background fits, not isolated
timing distributions or a guaranteed Kaggle improvement. Exact graph equality
is stronger evidence than the elapsed-time ratio. The reciprocal held6replay
is being launched next, with the same model, budgets and strict graph equality.
Receipt `trajectory-event-vectorized-replay-44b6-v1.json`; benchmarkSHA
`e273cc28892b1fcb6d053eb1447f0b01fcaf2a351173eab7912a27adc914d777`.
Replay sourceSHA`537b6165803335e12019af34e45939a218f1a01ce0d767002ff7a40fcf26ff8f`;
adapterSHA`f54b51c30fa0bac8983d647c55c124d7532ac128f4306721fee4b92e78f2b71b`.

Reciprocal replay is LIVE as session68610 at20:03UTC. First movie6bba_55b7eebe
already reproduced the exact graph in4.000s versus5.234s reference; dense second
movie is in progress. Observe this same handle, do not launch a duplicate.

## Reciprocal complete replay passed

Session68610 is TERMINAL success,297.281seconds total. Both held6complete graph
JSON hashes and every solver/edit record match the frozen controls exactly:

-6bba_55b7eebe:5.234s reference versus4.000s vectorized.
-6bba_57b7cc1e:490.109s reference versus287.391s vectorized,41.36%less elapsed.

All four preflight control graphs across both embryos now have exact full-movie
replay evidence,99transitions each and no fallback/cap. Single local runs during
background fits still do not establish a universal timing ratio, learned-head
performance, full60movie parity or Kaggle acceptance. No trained coefficient,
source fit, frozen evaluation successor or submitted notebook was modified.

The runtime experiment changed authoritative evidence this turn; it is not a
quality-score improvement. A prospective learned-head replay and deployment
acceptance remain necessary before installing the adapter in a candidate.

## 20:20 UTC: training-objective equivalence smoke passed separately

The private vectorized hinge adapter preserves the original training function
bytecode and substitutes only exact pruning in its private namespace. Original
training imports and both running processes are untouched. Seventeen combined
vectorized/reference tests passed in5.75seconds, including partial-division
constraints and six-step optimizer-state equality on a small synthetic problem.

Real-source benchmark95879 is TERMINAL success in9.938seconds. For each embryo,
use the fixed first source case plus the previously blocked fork8 case, in order
0,1,0,1. Start two ephemeral optimizers from the SAME frozen source-prior anchor;
compare original and vectorized losses/gradients and post-update weights/moments/
step after every update. All8comparisons match exactly, including the26,112and
24,913option cases. Training hinge timings5.195517s ->3.874854s,1.3408x speedup
in this bounded interleaved test, not a measured full-fit runtime guarantee.

Source-training labels were used only for this supervised functionality smoke;
no selection/validation labels, live intermediate checkpoints or output scores
were opened. No weights exported, live fit/resume contract changed, or quality
gain established. Receipt `trajectory-event-vectorized-training-v1.json`;
training-adapterSHA`333cc8a1a8e71203e95f86e842eeabf02ccc4c2cd40b5901bd0345ebd4bd63db`.

The same turn's native-cache reuse audit found matching image packets for only
867/5,780event transitions. Annotation-selected query coverage makes a naive
zero-filled feature join inappropriate. See native-cache-reuse findings; no
additional GPU run/image download or previously rejected neural fit was launched.

Combined original/evaluator/vectorized regression suite:42passed in10.16seconds,
session5199 terminal exit0. Live original trainingPIDs7572/43524 and successor
PowerShellPID35260 were reverified after testing; no restart. Diff check passed.
