# Weighted ranking improves missing-parent decisions, but does not pass

September 10, 18:33 UTC. Completed and verified the predeclared square-root
class-weighted full-candidate ranker. No GPU or new diagnostic/source/target
movie access. No final model exported or submission made.

| Twelve-movie correction-only LOMO | Physical | Original neural | Unweighted ranker | Weighted ranker |
| --- | ---: | ---: | ---: | ---: |
| Joint NLL, lower is better | 0.802871 | 0.667407 | 0.315919 | 0.362457 |
| Correct parent /10,754 | 9,514 | 9,835 | 9,963 | 9,891 |
| Correct absent parent /161 | 115 | 72 | 28 | 73 |

The new model gains 45 absent-parent decisions versus the unweighted ranker
but loses 72 correct parents. It still exceeds the original neural control on
both counts, with much lower NLL. Nevertheless, 73 is below the predeclared
115 absent-parent requirement: gate FAIL. Eleven movie NLL wins versus neural;
the sole regression is +0.0157899101 on 4f99ce20, within the unchanged 0.02 bound.
No post-hoc weight adjustment, diagnostic selection or acceptance relaxation.

The model's real-parent ranking ceiling is 9,982; it rejects 133 true parents,
selects a wrong source for 730, and creates 88 links on known-absent targets.
These counts identify remaining ranking and missing-parent errors; they do not
authorize treating unknown targets as negative training data.

## Execution and verification

- All 10,915 groups and 4,691,320 real/null choices retained, without sampling.
- Real both-class smoke: 57b7cc1e/frame31, 23,200 real candidates, 19 parents
  and one absence. Objective 107.898027 to 35.186469; 51 iterations/60 evaluations.
  Relative directional-gradient error 3.32e-12; exact model/prediction reload.
  Smoke total 9.188 seconds, optimizer 0.235 seconds.
- Full twelve-fold CPU-only wall time 597.297 seconds, below the 900-second cap.
  Per-fold weights use only the eleven fitting movies; no validation counts.
- Verifier reproduced every stored held-out decision and independently summed
  all weighted group losses, with exact training counts, control and gate
  replay. It did not refit the optimizers or claim global numerical optimality.
- Twenty-one relevant tests passed in 6.63 seconds; syntax and diff checks pass.
- Smoke41331, training10506, bound81531 and verifier98674 are terminal.

Smoke SHA256: 4c69554002fbf68ac3918ecf59aced3bbef87cf730aab275dfde309246b2db6d.
Result SHA256: dba19c4210c29941ca8056f49089ebac9d25eae8151c47931282af8366f5fe7c.
Actual folds: .biohub/cache/focus-balanced-candidate-ranker-v1/lomo.
Verification: focus-balanced-candidate-ranker-v1-verification.json.

## Independent error attribution: a physical veto is insufficient

The separate fitting-only calculation replays all old unweighted fold models.
Physical confidence rejects 612 true parents, including 434 that the new real-
parent ranking gets right. Retaining every physical rejection with that ranking
would yield only 9,550 correct parents, below 9,835 required. Do not run this
simple veto as if it were an untested promising fix. A hypothetical perfect
ranking under the same veto could reach 10,142, so this does not rule out every
future representation; it rules out the existing ranking/veto combination.

Audit SHA256: 695f5b3914f25a5d8fe1cac160648645f336f53aaba30a82820a129c67b16003.
Report: focus-ranking-abstention-v1-audit.json. No fitting, new evaluation pool
or submission was involved in that calculation.

## Next research and resources

The user-supplied LDA paper was reviewed while this run completed. See
lda-frozen-features-paper-assessment-20260910.md. A richer pair-representation
comparison is a distinct next hypothesis; no LDA quality result exists yet.
Do not continue this exact failed weighting configuration unchanged.

Fresh public-source screening found a newly updated explicit negative-time
graph exploit and excluded it. See public-refresh-20260910-1831.md and
research/public_notebook_exclusions.json. Nothing from those notebooks ran.

AWS credentials were verified working in the preceding user-status turn;
Antelume was stopped at that check. No instance, RSNA, shared environment or
GPU action occurred here. Last observed Kaggle quota remains 8.80 hours at
17:18 UTC, not a fresh reading; refresh before any GPU launch.
Previous status-only turn was no progress; this continuation completed new
training, verification, error attribution and actionable source review.
The full goal remains unfinished, with 0/5 newly qualified submissions today.
