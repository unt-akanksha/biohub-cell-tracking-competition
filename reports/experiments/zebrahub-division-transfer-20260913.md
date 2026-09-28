# External division transfer: execution record

September 13, 2026. Previous goal turn was progress: two terminal model screens,
independent source replay, verified backups, and a new source-coverage finding.
Neither previous screen qualified; their failures remain immutable.

## Optimization annotation audit

The new audit checked all 158 original optimization movies, including movies
with zero extracted examples. Role derivation follows the original allocator;
no source selection, audit or final-probe GEFF was opened. Every relevant GEFF
file was verified against the historical cache manifest.

44b6: 63 movies, 19 annotated divisions, all 19 extracted, five inference-eligible.
6bba: 95 movies, 72 annotated divisions, 67 extracted, 23 inference-eligible.
Five 6bba events were removed by distance bounds (overlapping rejection reasons:
two parent-distance, four sister-distance). No zero-example movie hides a fork.
The weak 44b6 source does NOT have an extraction bug that can add more labels.
Audit completed 30.641 CPU seconds, SHA256
`3666b4279bbb5aa8e8e053ad3e143c5664150a2e04b4ee06a96506d78c75ddab`.

## Distinct transfer experiment

The existing ZebraHub ZSNS004 external TRAINING split contains 425 two-child
lineage labels in 64 transitions / 4,009 parents. No external validation or audit
shards were opened. These are public tracking-derived labels, not independently
manual annotations. The [official imaging page](https://zebrahub.sf.czbiohub.org/imaging)
identifies Ultrack as the tracking method. The [organizer discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734330)
was retrieved in full through the Kaggle CLI on September 13 and confirms
competition use and no test overlap. The new website inspection does not
establish a data redistribution license; final provenance/license checks remain.

The dataset builder generated 7,152 examples, 425 positives, 213 eligible
positives and 284 eligible negatives, in 51.797 CPU seconds. It preserves all
true external pairs and selects at most two geometrically ranked non-child
negatives per parent, without model scores. Current 12/15 micron deployment
bounds and geometry>=3 mask are unchanged. There is no label/threshold search.

Frame alignment is explicitly fixed: external daughter patches originally
center on t+1 whereas every Biohub relational patch centers on t. Both domains
now use the same real image pair represented as [t,t,t+1]. Both use the existing
one-micron sampling grid, per-channel sample-standard-deviation normalization,
clipping and float16 quantization. Raw/difference velocity fields are explicitly
missing in both domains to avoid a trivial acquisition-domain indicator.
Context remapping matches full image-context recomputation exactly in tests.

A single L2=0.01 convex head will fit the fixed external encoder features, with
equal external/source-Biohub domain mass and no penalty sweep. Each model's
Biohub optimization, calibration and preprocessing exclude its opposite embryo.
Same source/transfer quality gates; all previous rejections remain closed.

## Staging snapshot: 21:19 UTC

Nineteen focused tests pass in 2.18 seconds. The 150-file, 264,482,640-byte
external/code bundle is transferring to the existing isolated Antelume directory.
No new encoder/head fit has started at this entry. The read-only pre-transfer
GPU process query was empty; 2,734,977,024 bytes of root storage were free.
No Kaggle GPU use, RSNA changes, shared installation or instance shutdown.

- External descriptor manifest:
  `ce4d395302104b0c5dbf447a53e411adb48ac4cfea7fbe02eac5765bae78b971`
- Frozen runtime contract:
  `3ef608e8153d6f141643a427d647c6792d3ff127fe9623512661cf6cdf5140ae`
- Transport archive:
  `62be2a7124066a9e9f10519340d2d26e03009fa252a375d23744a3d5184856ac`

This preparation is not quality evidence or a submission. Run the actual-image
smoke first, then at most one 15-minute sequential two-source screen.

## Terminal execution and independent verification

The real-image smoke PASSED in 7.928 seconds: four external plus four Biohub
optimization examples, strict encoder reload, exact saved-head prediction replay.
The full sequential screen completed 62.735 seconds, `rejected_source_selection`.

| Source | AP | TP above all negatives | Source gate |
| --- | ---: | ---: | --- |
| 44b6 | 0.750000 | 1 / 2 positives | Fail |
| 6bba | 0.706017 | 4 / 13 positives | Pass |

The joint gate FAILS; no opposite-embryo scores, competition test, or original
audit were opened. No selected model, threshold or training mix was revised.
The recipe is closed. This particular aligned, equal-domain frozen-encoder
recipe did not solve the weaker source and underperformed the prior 3-frame
Biohub-only head on the other source. Because temporal inputs, velocity handling
and training data changed together, this result does NOT isolate the causal
effect of external data or prove that all external-data training is unhelpful.

Independent CPU verification replays external/Biohub labels and eligibility,
equal-domain normalization, all source logits and source gates, removal of the
velocity-presence domain indicator, and the exact optimized convex objective.
Maximum absolute objective gradients are 4.584e-8 and 4.445e-8. Both heads
converged to the specified optimum; this was not a training-functionality fault.

All 17 smoke/full artifacts, including both external feature banks, source
feature banks, heads, predictions and terminal receipts, were copied locally
and independently matched the remote SHA256/size inventory. Cloud transport
archive was removed only after all 150 unpacked input files verified; the local
archive is retained. No model, feature bank or other project's file was removed.

Full-run encoder stages 37.051 seconds, head-fit stages 0.600 seconds, peak CUDA
allocation 466,944,512 bytes. These two job walltimes total 70.663 seconds;
today's six GPU-capable job walltimes total about 706.751 seconds (11.78 minutes),
NOT the billable uptime of the shared instance. Post-run GPU compute query empty,
allocated memory zero; Antelume remains running. Root free space 2,381,266,944
bytes. No Kaggle GPU use, RSNA process/file changes, packages or instance state
changes. No Biohub numerical job remains live.

- Full terminal SHA256:
  `8f7b43713c19f932448f8e8e137f4e26b04c11b96cfc22152399dd3c02e00ad2`
- Independent verification SHA256:
  `6edb882836181b371c3be5719b4c2a15080c4ccd30a9516ad2d20cb57686f8b3`
- Source 44b6 head SHA256:
  `789f145a41ebbe896d069f2effacae7eea04233512409c1b728b22664296de5f`
- Source 6bba head SHA256:
  `14b3b880d8dd8e4dfd7577639456deaa416bd3902f2962d1c14779d2bb946aa1`

Still no new qualified Kaggle submission. Do not use source-only numerical
improvements, or the earlier exposed 0.9493 four-movie diagnostic, as a new
leaderboard claim. Candidate/generalization and full-movie runtime requirements
remain unmet; the strong-submission objective is active and unfinished.
