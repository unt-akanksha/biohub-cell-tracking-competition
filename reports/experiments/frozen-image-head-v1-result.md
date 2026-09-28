# Frozen image encoder / small division head: rejected

September 13, 2026. This is a completed experiment, not a submission candidate.
Both source gates were fixed before fitting; the joint source gate failed.
Opposite-embryo predictions, original audit and competition test were not opened.
No leaderboard result is claimed. No rejected public repair was combined here.

## Measured result

| Source embryo | Selected L2 | Eligible selection AP | TP before first FP | Decision |
| --- | ---: | ---: | ---: | --- |
| 44b6 | 0.1 | 0.750000 | 1 of 2 positives | Fail |
| 6bba | 0.01 | 0.909557 | 9 of 13 positives | Pass |

The second source improves over the immediately preceding 74.7M fine-tuned
model's 0.824296 AP / 7 zero-FP TPs. The first source still cannot support the
required two-TP, zero-FP source threshold. These source-selection comparisons
are not embryo-transfer, complete-movie or independent leaderboard evidence.
The 6bba source threshold is 0.7779628422664069. It was never tuned on 44b6.

The encoder is the exact external-only microscopy warm start, frozen throughout.
The convex head has 1,347 parameters, versus the preceding 74,732,308-parameter
fine-tuned model. The four declared L2 fits converged on each source. Source-only
utility selected the recorded penalties. Fourteen tests pass, including analytic
gradient checks, fixed normalization, daughter symmetry, and saved-head replay.

The independent CPU verifier replayed source logits, tie-aware metrics, selected
penalties, source-only normalization, morphology, target/weight alignment, embryo
exclusion and movie separation. It verified all remote artifact hashes locally.
This is a verified rejection, not a training or transport failure.

## Remaining coverage gap

Source 44b6 supplies 19 optimization positives but only five satisfy the actual
inference geometry gate, plus three eligible negatives. Selection has two
eligible positives and two eligible negatives across three movies. The missed
selection division is `44b6_587a1e22-t019`, row 0. Its daughter separation is
14.6532 microns; eligible optimization positives span 8.9467 to 12.1400 microns.
Its proposed-parent distance is 9.9178 microns, beyond their 4.6320 to 7.1412
micron range. This is descriptive source-only coverage evidence, not proof of
causality, a relabeling decision, or authorization to retune selection gates.

Raw geometry legitimately contains missing-value NaNs in some 6bba examples;
the frozen existing normalizer emits explicit missing indicators and finite
model inputs. Earlier broad wording about finite prepared values should not be
read as claiming every raw geometry scalar is finite. Image patches, context,
normalized model features and predictions passed their finite checks.

Next useful work is broader, physically consistent division-positive coverage
and actual predicted-candidate validation. Do not keep varying seeds, penalties,
cutoffs or source splits to fit these four eligible 44b6 selection examples.
More GPU capacity alone does not establish a stronger submission.

## Runtime and recovery

Real-image smoke completed 7.872 seconds: exact encoder reload, a real source-
optimization-label head fit and exact saved-head predictions. Full screen
completed 16.521 seconds, including 6.734 seconds in encoder stages and 0.759
seconds in fitting stages; peak CUDA allocation 466,944,512 bytes. There was no
backbone optimizer or large new checkpoint. No runtime extension was used.

All 16 small-screen artifacts (models, feature banks, predictions, histories,
smoke and terminal receipts) are locally size/SHA-verified under
`.biohub/cache/frozen-image-head-v1-output`. The preceding fine-tuned pilot's
eight terminal artifacts, including both 299 MB models and the 1.184 GB rolling
optimizer/EMA/RNG checkpoint, are independently backed up under
`.biohub/cache/image-context-pilot-v2-output`.

At the 20:53 UTC read-only check, Antelume had no GPU compute processes and zero
allocated MiB. The shared instance remains running and still costs money while
idle. Root free space is 2,735,005,696 bytes. No RSNA files/processes, shared
packages, instance state or Kaggle notebooks were changed. Kaggle GPU usage was
zero. Four measured GPU-capable job walltimes today total 636.088 seconds
(10.60 minutes); that is NOT the billed duration of the EC2 instance.

## Receipts

- Frozen recipe contract:
  `a80e0e80bcf6103f78f10bdda658a737b00b90a839797c03f4941617f62db48f`
- Full terminal:
  `6206f476c9eec7fb335098e97b9e0df1db88874ee94ca691bd9d7937d91954bd`
- Independent verification:
  `cbf7827b1d666b4650c011bd5f66f9df2cd4ffc74303a8f1db8dd01622c57706`
- Source 44b6 head:
  `f6c0feeba02cadcc30e044d83244b106cfa83115d0ad4490cbd7da4dd6875d6f`
- Source 6bba head:
  `98fa0a0dbdedb1ef153004193205b2efb1f218aaeca1908b4e46420a748a5504`
- Preceding image-context v2 terminal:
  `90953144c2f6c0f33521014776c725aeea552e578bc0a0e0b154499c4a5ea85a`

No live Biohub job, new Kaggle submission, or newly qualified submission remains
at this handoff. The user's strong-submission goal is unfinished.
