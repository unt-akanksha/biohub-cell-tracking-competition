# Past-motion additive head: rejected

Completed September 13, 2026. The frozen recipe in
[the design](comoving-division-head-v1-design.md) was not extended or tuned.

| Source embryo | Fixed control AP / zero-FP TP | Motion AP / zero-FP TP | Motion source gate |
|---|---|---|---|
| 44b6 | 0.750000 / 1 | 0.833333 / 1 | Fail: requires two TP |
| 6bba | 0.909557 / 9 | 0.863293 / 4 | Pass |

The joint source gate fails. Opposite-embryo scores were not opened; no GPU,
inference change, promotion, or submission followed. This closes this recipe,
not the broader hypothesis that motion can help tracking. The optimization-only
geometry audit was mixed and does not justify replacing the existing gate.

Audit took 22.422 CPU seconds, feature preparation 14.484 seconds, fitting
0.609 seconds. Six focused tests pass (1.42 seconds). Independent verification
checks all 2,480 candidate ID triples, unchanged original features and targets,
source-only normalization, saved logits, gates, and convex stationarity.

Receipts (SHA-256):

- Geometry audit: `d0d27781e48beb3278dea9d4a3b2b76762a1beedd0d951ef541b9c2ecb0509bc`.
- Data manifest: `dfd56ed703fa9189079bb37714d6d25fd569f5e3a31c01c5a8e8879ddb6435e7`.
- Terminal result: `30418e78f4288c18e8cc6be6fb324c54d28f910923414bf26aba6825815cd41f`.
- Independent verification: `aac3af2a6dabcf69d2822e8d4cd599edce20fa0264345593c2a964be0608ac3c`.

All outputs are local under `.biohub/cache/comoving-division-head-v1-output`.
No live numerical job remains from this experiment. Antelume was not accessed
by this CPU experiment; its current occupancy and billed uptime are not inferred.
