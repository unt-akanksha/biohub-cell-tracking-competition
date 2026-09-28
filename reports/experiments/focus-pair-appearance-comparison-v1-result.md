# Frozen appearance comparison: completed, verified, neither arm promoted

September10,20:15UTC. The full72D numerical recovery completed all12 folds in
924.812s CPU-only wall time. Independent replay completed422.953s. The original
unsuccessful solver and all completed records remain unchanged. No GPU used for
these fits/replays. No final all-fit model, diagnostic/source/target evaluation
or submission follows either failed arm.

| Same12 correction-held-out movies | Prior weighted | LDA9D | Full72D |
| --- | ---: | ---: | ---: |
| Pooled unweighted NLL, lower better |0.3624569|0.3427010|0.3494529|
| Correct parents /10,754 |9891|9880|9854|
| Correct absent /161 |73|84|84|
| Missing-parent requirement115 |FAIL|FAIL|FAIL|

Full features improve per-movie NLL over LDA on9/12movies but are worse when
pooled over all known groups. The difficult57b7cc1e movie regresses from
LDA0.857580 to full0.906262. Do not select an arm by counting movie wins alone.
Both new arms pass the other four unchanged gates but fail the missing-parent
requirement. No threshold, class-weight, gate or original physical control was
adjusted. These are correction-only feasibility screens, not independent
encoder/embryo validation or official tracking/leaderboard scores.

The verifier rebuilt every fold's scaler/Fisher statistics from its11 training
movies, replayed each original streamed training objective and gradient,
recomputed the complete initial Hessian and bound-preserving coordinate map,
checked saved optimizer states and all25iteration checkpoints, and independently
replayed held-out group losses/choices, comparison controls and acceptance.
It did not rerun any optimizer or open diagnostic/source/new target movies.

Full result SHA256:
c32589575cd1d225b63c93bbcea8e9f332f629798e499b767c90a22549ee3919.
Wrapper SHA256:
5364d3d0231fe86d43843edce243cbba754d7bb4d6d322397969575df579d7f6.
Verification SHA256:
370fb67ec26f99842be51ebd89b26a55494b96959a88366805a328b5512e7450.
LDA verification remainsc3045123571fa1daccf23f3cd784d751ca240f9bf384b942a1a5562337f704f4.

## Distinct next branch, not deployment of a failed arm

A fixed nonlinear tree residual can change real-candidate ranking through
appearance/motion interactions; this differs from the retired presence-only
trees and linear corrections. It starts from a matching fold-only full linear
baseline, not a globally fitted model. No claim that it will generalize.

The small original20group/23,220choice real smoke passed:100depth3trees fit in
1.719s, loss13.705099 to3.855849,18 to20 correct training decisions. This is
training functionality only. Native/portable prediction difference0, real
gradient finite-difference error3.60e-11, all candidates and input arrays intact,
four native checkpoints and full native/portable model saved.18 targeted tests
passed2.95s. Smoke receipt41fd6c6e9d8ea8cfa67a0b6d6d4917958e430fd543eaaebf1fac3e3e70836e2c.

At20:15 the full first-fold fitting-only profile is live in session87429,
with900s wall cap and6GiB peak-working-set guard. It uses exactly11 fitting
movies and does not load/score the excluded movie. No full12fold tree quality
run yet. Local dependencies are isolated from graph-analysis and shared AWS.
All earlier fitting/verification/smoke/install handles are terminal. No GPU,
cloud or RSNA changes; quota8.22h19:17 remains historical, not current balance.
Goal unfinished,0/5 newly qualified submissions.
