# Nonlinear pair-ranking screen: verified rejection

September 10, 2026 UTC. All 12 folds and the independent verifier are terminal.
Training took 3,478.985 seconds on local CPU. Verification took 314.968 seconds
(316.875 seconds including controller), peaking at 3,133,399,040 working bytes.
No GPU, new diagnostic/target labels, full-data model export or submission.

| Arm | Pooled NLL | Correct parents / 10,754 | Correct absent / 161 |
| --- | ---: | ---: | ---: |
| Original physical control | 0.8028714831 | 9,514 | 115 |
| Original neural control | 0.6674070155 | 9,835 | 72 |
| Fixed 100-tree candidate | 0.3457728923 | 9,848 | 83 |

The unchanged gate fails `absent_not_lower`: 83 is below 115. Other predeclared
conditions pass. Neither a lower loss nor the small parent-ranking gain
authorizes overriding the failed missing-parent check. The previous LDA arm
also has better pooled NLL and more correct parents than this tree arm; do not
extend the same tree run or promote it as a submission component.

Every fold was replayed against all fitting choices, checkpoints, native and
portable inference, and independent held-out logaddexp/argmax calculations.
Native/portable held-out maximum error was zero in all folds. These are
held-out correction-head folds, not proof of an independent underlying encoder.

Receipts:

- `focus-pair-tree-lomo-v1-result.json`:
  `47dd9b60ff80b621f463cd2065dfc2283a8c9599a2e608cc0c02fda407c0b28a`.
- `focus-pair-tree-lomo-v1-verification.json`:
  `fb9d347528607754bafbd29b4800d77d02a90476aa212beb1aa58ed239060d05`.

The separate history-head optimizer smoke passed on 20 original fitting groups
with 23,220 choices (gradient error 1.10e-11, Hessian error 2.97e-10, 45
iterations, exact reload). That is functionality only. Further history-data
staging and full history-quality training are paused in favor of the user's
requested public-baseline enhancement. No background tree/history job remains.
