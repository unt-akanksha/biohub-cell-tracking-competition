# Isolated motion-field spatial averaging

V2 small GPU functionality test completed and verified. No complete-movie
motion-field accuracy test has run yet. No submission authorization.

Use the staged **v2 three-arm probe**, not the earlier v1 staging. V2 first
compares native standalone flow with an opt-in shortcut that avoids evaluating
the neural network when its coefficient is exactly zero. It requires identical
actual node coordinates, edge pairs and official counts, plus execution counters
proving the neural forward calls were skipped. The third arm adds flow D4 to
that verified execution policy. Thirteen local CPU/build checks pass; new Torch
and actual graph parity tests remain unexecuted until the GPU probe runs.
Cold versus cached three-frame timings must not be advertised as a speedup factor.

## Verified v2 result

`biohub-backward-flow-spatial-tta-probe-v2/1` completed in71.295s;29 GPU-environment
tests passed in4.85s. Native and zero-weight-shortcut runs produced identical
269nodes/160edges and identical official counts:14TP,5FP,2FN,division0TP/1FP/0FN.
Execution counters confirm2 native neural calls versus0 shortcut calls.
All frozen flow hashes match the original e82b7255 tensor hash.

FlowD4 retained all269 coordinates exactly, produced161edges, and scored
14TP/6FP/2FN withdivision0TP/2FP/0FN. Maximum mean absolute flow change was
0.0314999 microns. This is functionality evidence; tiny training counts worsened,
so no claim of accuracy improvement. Verified report:
`backward-flow-spatial-tta-probe-v2-result.json`.
Notebook SHA240e10391ba83992f4ecde60e3b3cdf463d2cdc9c33863a6d34f320772477a8c.

The retained parent detector D4 misses999 annotated edges despite both
endpoints being matched, compared with371 misses requiring changed detections.
This motivates a separate association hypothesis, not proof that averaging
motion will recover those edges.

Freeze parent76f7 detector D4 and embedded flow3006; use native detector
features/logits and the original cutoff. Average the learned backward flow
over eight XY dihedral views. Each field is inverse-transformed both spatially
and in its physical ZYX vector basis. Keep FP32 averaged fields, original
variance, neural0/spatial1/null-4.5, one-parent/two-child rules and2048-node
abort. No graph pruning, threshold fitting or additional target access.

CPU geometry tests cover every rotation/reflection using independently
transformed point displacements. Real Torch tests additionally check rectangular
spatially varying fields, vector sign restoration, directional-bias cancellation,
CUDA device behavior, and evaluation-only installation.

The prepared three-training-frame probe compares parent D4 plus native flow
against parent D4 plus flow averaging. It must verify identical actual node
coordinates, strict checkpoint/flow hashes, unchanged flow weights, nonzero
executed averaging, finite graphs and GEFF round trips. The shared smoke helper
gained an optional flow callback; all already launched notebooks remain frozen.

Only successful real functionality evidence can justify a complete-movie test.
No claim is made that scalar heatmap transforms alone are valid for vectors,
that the averaged flow is more accurate, or that this is a submission candidate.
