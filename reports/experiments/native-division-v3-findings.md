# Native joint-division experiment v3: screening rejected

The experiment completed; no new graph edits, ensemble or Kaggle submission
were authorized. The existing trajectory-motion submission is unchanged.

## What was actually trained

The previous four native correspondence encoders were frozen. We fitted four
new regularized linear **joint parent/two-daughter event** heads, one per
source embryo and encoder origin, plus two geometry controls. Each fit used
2,000 full-batch updates. The transformer-origin head uses that model's learned
residual encoder; it does not run the old candidate transformer or parent head.
These are newly trained event classifiers, not copied public predictions, and
not four new end-to-end large-network training runs.

The data extraction added 38 safe positives to six existing examples. Counts
are 44 optimization positives and 2,305 negatives, with only seven selection
positives and 731 negatives. All 112 annotated optimization division events
were considered, but only 44 yielded safe image-proposal triplets. No excluded
pilot movie, sealed audit or unassigned movie supplied training labels.

## Frozen screening result

Thresholds were set once from source-selection negatives. Zero observed source
false positives are therefore a calibration condition, not independent proof
of precision. Source selection additionally required >=50% division recall and
better balanced NLL than geometry. Passing heads were evaluated on the opposite
embryo without changing weights, thresholds or selection roles.

| Source | Head | Source TP / positives | Source FP | Balanced NLL | Opposite check |
| --- | --- | ---: | ---: | ---: | --- |
| 44b6 | Geometry control | 0 / 1 | 0 | 0.80754 | Not opened |
| 44b6 | CNN-origin | 0 / 1 | 0 | 0.54857 | Not opened; source rejected |
| 44b6 | Transformer-origin | 0 / 1 | 0 | 0.35527 | Not opened; source rejected |
| 6bba | Geometry control | 3 / 6 | 0 | 0.18220 | Comparator only |
| 6bba | CNN-origin | 4 / 6 | 0 | 0.10699 | 0 / 1 TP; 0 FP; NLL 1.28714: rejected |
| 6bba | Transformer-origin | 5 / 6 | 0 | 0.08742 | 0 / 1 TP; 0 FP; NLL 1.42862: rejected |

The source6 heads add real source-selection signal over geometry but do not
meet the cross-embryo gate. Neither pair qualifies for an ensemble. The single
usable opposite-embryo positive is too small to establish broad generalization
claims, positive or negative; nevertheless the declared screening gate failed.
Do not weaken that gate, tune on the missed held-out event, or send these heads
through a target-pilot threshold search. This candidate is closed.

## Functionality and resource evidence

- Local sparse-label/identity tests: 3 passed; Torch feature test module skipped
  locally because that analysis environment has no Torch.
- Antelume lacks pytest. Equivalent Torch assertions were run directly without
  installing into its shared environment: known geometry, exact daughter-order
  invariance, constant-image statistics, empty input and nonfinite rejection pass.
- Native feature smoke: 20 image patches / 16 triplets; 4.598 seconds. Exact
  encoder reload and daughter-order checks passed for both source folds.
- Head smoke: both classes handled, single-class input rejected, exact saved
  checkpoint reload, 2/2 synthetic positives and zero false positives.
- Full feature extraction: 4,900 image patches, 3,087 triplets per reciprocal
  source fold, 8.835 seconds.
- All six classifier/control fits and screening: 14.916 seconds. No overnight
  run is needed for these frozen-feature heads.
- Four result groups, all feature arrays and six classifier checkpoints were
  copied locally and every file hash verified: 46,366,568 bytes. GPU jobs ended.
  RSNA, other processes and shared Python environments were untouched.
- After backup and a zero-live-GPU-process check, the redundant
  `/dev/shm/biohub-native-correspondence-v2-full` dataset was removed: 1,218,965,320
  file bytes, recoverable from the local native-correspondence-v2-data cache.
  All 2,103 packets plus RESULT/PROGRESS logs were verified locally and remotely.
  The first cleanup guard correctly refused an unbacked PROGRESS.json; it was
  copied and hash-verified before retrying. No broad cache drop or GPU reset.
  At 06:19 UTC GPU memory was 0 MiB, no compute processes, available system RAM
  14,806 MiB, and /dev/shm 190 MiB used. AWS remains running and billing.

## Reproducibility

Frozen contract:
`e263730431b25c681218f21d2af1673f6f5be3c85f33266aa86986a0904a3e96`.
Merged data manifest:
`7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06`.
Full feature result:
`50173b81c2798f8524a14ff9b8d11752a7ea714694184f26e943846db5722130`.
Weights and full metrics are in `native-division-v3-head-full-result.json`;
verified recovery records are in `native-division-v3-harvest.json`.

## Next measured question, not a queued run

The immediate upstream limitation is sparse usable divisions: only 39.3% of
available optimization division annotations become image-centered examples.
Audit source-optimization proposal coverage by parent, first daughter and second
daughter before proposing another architecture. Separate missed image proposals,
ambiguous one-to-one matches and displacement exclusions. Do not insert GT
centers into an inference candidate or infer negatives from unlabeled cells.
Any proposal change needs a separately frozen source-only protocol, measured
inference cost and complete-movie patched scoring before promotion.

This experiment does not establish a stronger submission or a top-five result.
