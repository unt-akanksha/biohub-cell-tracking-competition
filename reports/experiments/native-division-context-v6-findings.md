# Temporal division context v6: rejected

The R2 run completed all 12,000 planned successful updates across eight paired
fits in 324.73 seconds, including 250 recovered updates. No expert passed both
source and opposite-embryo gates. No ensemble is eligible, and no new submission
is authorized by this experiment.

## Evidence

The unchanged 3,108 image-derived triplets gained neighboring-frame image crops
at t-1 for parents and t+2 for daughters. Boundary observations were masked;
labels, roles and candidate locations were preserved. A 9.67M-parameter model
was trained with context on/off for each of two source-specific encoder origins
and each of two source embryos. The old association heads were discarded: the
transformer-origin label denotes encoder initialization, not a transformer event
head. The mixed-source functionality-smoke weights were never reused.

| Source / encoder origin | Control source TP | Context source TP | Chosen arm |
| --- | ---: | ---: | --- |
| 44b6 / CNN | 0/1 | 0/1 | None |
| 44b6 / transformer | 0/1 | 0/1 | None |
| 6bba / CNN | 0/6 | 0/6 | None |
| 6bba / transformer | 4/6 | 0/6 | Control |

All source operating points had zero false positives by the frozen calibration
rule. The only selected arm, 6bba transformer-origin without temporal context,
failed opposite-embryo screening: zero of one true event recovered and two false
positives. Its opposite balanced NLL was 2.75191 versus geometry 0.49269.
No losing arm was evaluated on the opposite embryo. Selection contains only
seven positive events and does not establish reliable population performance.

Some learned probabilities saturated so that max-negative probability + 1e-4
exceeded one, resulting in zero recall. This is a rejection under the frozen
operating rule, not permission to relax the threshold after seeing validation.
Near-zero training losses and failed screening are consistent with overfitting;
larger ensembles of these weights are not justified. Do not repeat this small
supervised data recipe with more capacity or tune against these failures.

## Numerical recovery and preservation

R1 stopped after 428 successful updates. Exact recovery from update 250
reproduced an FP16 gradient overflow at the same update with finite loss.
R2 skipped overflowing updates and reduced the loss scale, without changing
the scientific protocol. Diagnostic weights were not used for selection.

All context data, smoke outputs, R1 checkpoint and diagnostic, and R2 final
weights/history/resume state were recovered and SHA-256 verified locally.
R2 backup contains 425,746,956 bytes; archive SHA-256:
`52db31ee28e3f5286b403466938063d4e95247bfb0f7abb6a820a3c6a48bc35d`.
The training contract is
`36d024261fa71f38ff6b273c3fcb94ec48fff0272cfd7f4d8374e0036f796150`.

Authoritative artifacts:

- [R2 result](native-division-context-v6-training-full-r2-result.json)
- [R2 backup receipt](native-division-context-v6-training-harvest-r2.json)
- [Frozen training design](native-division-context-v6-training-design.md)

The pre-existing trajectory submission is unchanged. This result is not a
leaderboard improvement, a complete-movie tracking score, or a top-five claim.

At 07:53 UTC, after a passing dry run and repeated local/remote hash and process
checks, removed 785 backed-up terminal RAM files and redundant transfer archives
(1,447,100,031 bytes) from six exact v6 directories and three exact tar paths in
`/dev/shm`. All remain recoverable locally. No system cache drop, GPU reset,
foreign-process termination, root-disk cleanup or instance shutdown occurred.
See [cleanup receipt](native-division-context-v6-memory-cleanup.json).
