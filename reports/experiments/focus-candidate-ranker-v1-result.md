# Direct parent ranking improves, but missing-parent acceptance fails

September10,18:08UTC. Completed the fixed candidate-supervised motion/appearance
ranker, not another residual-density or presence-only fit. Its features and
parameters are original to this experiment; no public checkpoint, output or
leaderboard setting was adopted. Cached image features come from the existing
owned encoder. This is not a new independent end-to-end model or submission.

## Functionality and complete data

Verified all12 fitting movies, all99 transitions per movie, and exact known
labels against the previous neural summaries. The exported choice cache has
10,915 supervised targets and4,691,320 real-candidate/null rows,338,386,280bytes.
No source candidate was truncated or sampled away. Unknown targets are ignored
only by the supervised loss; the label-free feature function covers all targets.
Original physical-null replay recovers115/161 and meanNLL0.96162087223.

Largest-packet smoke:6bba_57b7cc1e,sourceframe31,23,200 real candidate rows.
Finite-difference directional-gradient error8.06e-9; objective107.019->30.049;
53iterations/64evaluations in0.094seconds. Saved-model decisions replay exactly.
Extraction plus smoke87.938CPU seconds. This smoke establishes functionality,
not held-out quality. Full LOMO followed only after the smoke completed.

## Twelve-movie leave-one-out result

| Pooled correction-validation measurement | Physical control | Original neural control | New candidate ranker |
| --- | ---: | ---: | ---: |
| Unweighted joint NLL | 0.802871 | 0.667407 | 0.315919 |
| Correct parent /10,754 | 9,514 | 9,835 | 9,963 |
| Correct absent parent /161 | 115 | 72 | 28 |

NLL improves on all12 movies, and128 additional parent links are correctly
selected versus the neural control. However, missing-parent correctness drops
to28/161, far below the fixed requirement of at least115. The gate FAILS.
No final all-fitting model, diagnostic evaluation, source evaluation or
submission was authorized or produced. Do not relabel the failed model as a
qualified candidate or relax its gate after seeing the results.

This establishes a useful ranking improvement with an unacceptable abstention
trade-off. Further work must preserve absent-parent reliability while retaining
the ranking gain; repeating this unchanged configuration is not justified.
Do not claim these NLL values are tracking or leaderboard scores. All fitting
movies were used by the original encoder, so LOMO covers the correction only.

Full LOMO254.656CPU seconds under the900second cap. Host verifier replayed all
saved held-out decisions and control metrics, complete choice/label identities,
training objectives/counts, source hashes and screening arithmetic. It did not
claim an independent optimizer refit or a proof of global numerical optimality.
Seventeen related tests pass1.23seconds; syntax checks pass. All processes
terminal (extraction41809,LOMO71461,verifier4475).

Data/smoke SHA256:
c0376562166ca50ffdb12e9f79aff7bc867a217c5f69104454e723e45c3b8ccf.
LOMO result SHA256:
1682bf7167f5119bad272a26f035b15a1cfe25ae4c4ce6db7cc4a13a30446c92.
Verification SHA256:
6044fabdc1ada181d6a7fd6d583e7998fc4ad474bc21936734376b055de864cf.
Actual choice arrays/fold models: `.biohub/cache/focus-candidate-ranker-v1`.

## Resource and access correction

The user refreshed AWS credentials at17:46UTC. EC2 inventory now succeeds;
the latest actual check at17:57UTC reports Antelumei-0d12195df0d3558f3 STOPPED,
g5.xlarge, no publicIP. Thus the old ExpiredToken blocker is resolved; current
cloud availability is stopped-instance state, not expired credentials. No AWS
instance action, RSNA/shared-environment mutation or GPU work occurred this turn.
Last observed Kaggle quota8.80h at17:18 must be refreshed before any launch.

Previous substantive goal turn was progress (verified full-source uncertainty
test); the intervening user-status reply established refreshed access and the
stopped instance. This turn is progress: complete data, real smoke, full LOMO
training and verification. Goal unfinished;0/5 new qualified submissions today.
