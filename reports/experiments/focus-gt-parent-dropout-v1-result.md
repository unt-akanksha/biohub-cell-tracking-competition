# Exact-GT augmentation audit and serialized replay passed

Fresh official matching on all12 fitting movies exactly replayed all1,188
original frame-pair label sets. Exact GT parent centers then defined the fixed
7um source-candidate exclusion neighborhood, with unchanged parent identity
selection. All newly null labels have no retained candidate within7um of their
annotated parent. Original unknown targets remain unknown; target arrays and
inference detections are unchanged. These are training-only candidate sets.

Generated1,121 nonempty-source augmented pairs with1,121 synthetic null labels,
161 natural nulls and9,633 retained parent labels. The previous14um construction
retained9,539 parent labels. Independently reloaded all12 sidecars and original
NPZ packets; all1,121 augmentation decisions, output identities/label hashes,
supervision totals and physical-difficulty statistics replayed exactly.

| Synthetic null difficulty | Previous14um | Exact-GT7um |
| --- | ---: | ---: |
| Wrong physical parent decisions | 1 / 1136 | 31 / 1121 |
| Mean null NLL | 0.004479 | 0.076444 |
| Median best-parent minus null logit | -31.19583 | -11.75998 |

The prospective data-feasibility gate passed. However,97.23% of these synthetic
null cases are still physically solved, versus71.43% of natural fitting nulls.
This does not establish a matched difficulty distribution or improved neural
accuracy. Only a small separately frozen real training smoke is authorized by
this data result; model promotion still needs unchanged real-data diagnostics
and complete-movie tracking validation. No GPU, diagnostics/source-selection or
target data were used in this audit. No new model or submission was produced.

Audit61.844CPU seconds. Eleven relevant tests passed2.83seconds.
AuditSHA: `d27222f2c768105a149226b2b864fe3931a3f1ff5f4f4a415cd079c87c4afd2f`.
Metadata/GT-center sidecars: `.biohub/cache/focus-gt-parent-dropout-v1`.
Verification: `focus-gt-parent-dropout-v1-verification.json`.
`scripts/verify-focus-gt-parent-dropout.py::verify()` returns the verified receipt
and a complete ordered bundle for a future offline worker. If embedding that
bundle, hash the exact emitted bytes; do not repeat Windows newline normalization
mistakes. Preserve all old staged/launched artifacts unchanged.
