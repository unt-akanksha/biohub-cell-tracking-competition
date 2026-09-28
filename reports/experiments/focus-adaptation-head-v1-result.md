# Head adaptation completed; diagnostic feasibility FAIL

Kaggle `biohub-focus-adaptation-head-v1/1` completed800 fixed updates.
Worker93.532 seconds. Four-step real gradient/frozen-weight/checkpoint smoke
passed. Frozen encoder/detector tensor digest stayed unchanged. No detector
rerun, new raw nodes, target/source-selection access or submission.

Host recovery and verification completed: actual runtime bytes, step4/800
checkpoint hashes, immutable notebook/spec and diagnostic counts all match.
The frozen feasibility gate was independently recomputed and fails unchanged.
Launcher136.219s; worker resultSHA8d8407da.... Verified receipt is
`focus-adaptation-head-v1-result.json`. Fresh quota10.52h; no Biohub GPU job
or follow-up remains running/queued. AWS utilization is still unverified.

All three arms use exactly2,645 known-parent and27 known-absent targets on
the four diagnostic movies, excluded from this optimizer but included in
the original encoder/head training. These are training-domain diagnostics,
not official tracking scores or independent validation.

| Diagnostic arm | NLL (lower is better) | Correct parent | Correct absent |
| --- | ---: | ---: | ---: |
| Calibrated physical-only | 0.333914953 | 2500 | 18 |
| Initial owned neural + physical | 0.219008899 | 2585 | 12 |
| Adapted owned neural + physical | 0.217096080 | 2588 | 11 |

NLL and parent-count requirements pass, but missing-parent correctness fails
against both controls. Do not relax the gate or extend this checkpoint to
source tracking evaluation. No model is promoted. This does not establish
superiority to any public solution.

Final step800 checkpointSHA256:
`be0128e1ba96ec172b06cfe4f6dca058037296b4f4c8ac866e12630662e8774f`.
Final model tensorSHA256:
`75c7f4ae730c6418306ad837d5fe38d8c737faf7ca438c82a73b0e2b5431d16d`.
Fitting used334 supervised pairs, diagnostic377; unknown labels ignored.
The complete raw and feature caches remain reusable for a separately declared
training intervention without repeating the35.5-minute FOCUS detector run.

Next hypothesis, not an active run or established explanation: the fixed-null
association model needs better missing-parent discrimination. Evidence does
not justify simply training longer, relaxing the posterior threshold, or
assuming that larger models/ensembles will solve this failure.
