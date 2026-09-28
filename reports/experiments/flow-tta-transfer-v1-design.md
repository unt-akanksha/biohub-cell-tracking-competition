# Frozen flow-D4 transfer diagnostic

Source validation completed in 1048.855 GPU-launcher seconds and passed its
predeclared gate. Score 0.6321840591 versus 0.6298395328; raw edge Jaccard
gain 0.0021527269; node recall unchanged; five of eight movies improved.
Counts: 5462 TP / 1199 FP / 1348 FN, versus 5440 / 1192 / 1370;
division 3 TP / 268 FP / 8 FN, versus 2 / 265 / 9. Source report SHA-256
`624d0fce503dac81b93e6f2ff6f7113ad62ec4f3999c7200dc943a1afe64206f`.
The source CPU controller completed successfully and must not be restarted.

The transfer candidate is unchanged: parent76f7 detector D4, embedded flow3006
with eight inverse-aligned XY vector/grid views, FP32 flow mean, and verified
zero-neural shortcut. No threshold, variance, null logit, topology or detector
weight changes. Checkpoint, source report, probe and split are hash-bound.

Only the four already-exposed44b6 movies are included, in their original
audit order. Exact candidate coordinates must match the original detector-D4
audit graphs, whose manifest SHA is
`cbc49f2ba442b9ff1f8cfc276e7e6291d852bc22929f362e540bae486bd84ac5`.
This is a transfer diagnostic, not fresh independent confirmation. No new
target movie or submission is authorized by this run.

Before seeing transfer results, retain the existing transfer criteria:
positive aggregate score and raw-edge Jaccard deltas; mean recall loss <=0.005;
at least three of four adjusted-edge movie scores improve; every movie loss
<=0.02; worst-movie loss <=0.01. Use the unchanged patched official scorer
with complete-movie micro-aggregation and per-movie/embryo reporting. No
post-result coefficient sweep or threshold relaxation.

73 relevant regression checks pass (72-suite run plus the added actual-CLI
regression; the final16 transfer tests pass after repairing the import path).
The CLI failure occurred before any staging directory was created or job
launched. Frozen staging:

- GPU `biohub-flow-tta-transfer-v1`: notebook SHA
  `76e3075c9b6868cc47408f8d6be4d7a4551000d2eeb448ac94776d0890668b80`.
- CPU `biohub-flow-tta-transfer-scoring-v1`: notebook SHA
  `fa71524b0bc23ed23f5cf4ac23c91a780ccde0d233ba0d917b1bb1b9ee865040`.

One-hour total GPU declaration, earlier internal watchdog, Internet/TPU off.
The previous GPU job is authoritatively COMPLETE. Fresh quota is required
again at launch; the staging observation was 12.30h. AWS remains untouched.

The separate cached-FOCUS/owned-flow probe is now staged but unlaunched.
31 geometry, real cached-node GEFF round-trip, CLI and receipt checks pass.
It is secondary to this measured positive source result and cannot overlap
the active transfer GPU run. See focus-owned-flow-probe-v1-design.md; there
is no new accuracy claim.

Launch receipt: `indarkarhana/biohub-flow-tta-transfer-v1/1` accepted after
fresh12.30h quota and authoritative previous-GPU COMPLETE checks. Live logs
confirm offline dependency setup and inference startup; no error observed.
One-shot CPU controller PID40824 started08:52:27UTC and is live.40 focused
tests cover the transfer contract, comparison safeguards and hash-frozen
CPU-only follow-up. It cannot launch GPU jobs, submit, rebuild stages or open
new target movies. Its maximum duration is two hours; do not start a duplicate.
