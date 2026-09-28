# FOCUS residual calibration v1: measured gain, promotion rejected

Completed September 10, 2026. CPU elapsed 75.704s; no GPU use, new target
access, or competition submission. The separate design and executed sources
remain frozen. Eight focused model/attribution tests passed again after execution.

Fit a pooled diagonal Gaussian to 928 uniquely matched ordinary next-frame
links from two original training movies only: 6bba_f1fde7e0 (438) and
6bba_23af9eeb (490). No source-eight labels entered the fit. Parameters were
persisted before candidate source evaluation. All eight candidate arrays were
saved before their GT was opened. Fresh controls reproduced prior counts.

Residual mean ZYX in micrometres: [0.6847597561, 0.4953975872, 0.6188701264].
Variance: [2.5738654939, 2.3743091105, 3.6901129377] square micrometres.
The native-voxel variance floor did not activate. No threshold sweep, trimming,
node changes or division-count manipulation. Existing null cost, posterior
threshold and graph-degree limits remain unchanged.

| Fixed-eight source measure | FOCUS owned-flow reference | Calibrated |
| --- | ---: | ---: |
| Combined score | 0.7753325939363426 | 0.7805420622074996 |
| Raw edge Jaccard | 0.7937315831770694 | 0.7987813134732566 |
| Mean node recall | 0.9643973728718713 | 0.9643973728718713 |
| Division TP / FP / FN | 3 / 186 / 8 | 3 / 154 / 8 |

Five movies improve versus FOCUS flow, three regress. Seven improve versus
the original parent. Movie 6bba_67ebd073 improves by 0.01341083 versus FOCUS
flow but remains 0.02471136 below the original parent: the frozen maximum
loss of 0.02 is violated. All additional FOCUS-flow comparison conditions pass;
the original source gate does not. Do not relax it retrospectively.

Decision: retain the gain as component evidence, not a promoted candidate.
FOCUS pretraining overlap is unresolved. These are repeatedly exposed source
development movies, not new embryo-held-out evidence or a leaderboard estimate.

Related completed attribution: of 613 FOCUS-flow missed links with detected
endpoints, 540 are beyond the fixed motion gate, 72 fail posterior competition,
and one is topology-blocked. Another 271 GT links lack a detected endpoint.
This supports investigating motion errors, not oracle edge insertion or
arbitrary distance-gate expansion. Existing image-registration code has been
identified for review; no registration experiment is staged or running.

Authoritative result: `focus-residual-calibration-v1-result.json`, SHA-256
`b765d6105b48f31969bf149342ba7f87ae846f46bc1c5c9831fcd04fede0526e`.
Fit: `focus-motion-residual-fit-v1.json`.
Attribution: `focus-source-motion-error-attribution.json`, SHA-256
`4af7b5d47003513c2c6fadf838cff63dea566968d54142e85a7fd0574610da34`.
