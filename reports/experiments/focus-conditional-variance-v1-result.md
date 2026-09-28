# Conditional uncertainty: fitting screen passed, tracking test failed

September10,17:43UTC. Completed one fixed CPU-only heteroscedastic diagonal
motion model. Conditional means, input coordinates and graph rules were unchanged.
No hyperparameter or source-specific threshold search was performed.

## Fitting-only result

All14 movie-held-out uncertainty fits and the final all-fitting model were
independently recomputed from verified arrays. Proper Gaussian NLL improved
5.741789848 to5.698240760 on11,661 ordinary matched transitions;9/14 movies
improved. Worst-movie NLL decreased and mean/MSE remained exactly unchanged
(8.516682181um2). The fixed component gate passed. CPU screen1.969seconds.

Final model SHA256:
9ee9937ac60f0abc9d6127ac07e9696399cbbbe0e94eb9e2e6498a35fee27e25.
Training result SHA256:
f23d0004527b06d0401083a37ae0c9312c5c3b1646633ad1c74a9d2decc52a85.
Training verification SHA256:
7963142a3cbe1bedaa4b6be96c01d16166d9c9a74020f126902072e3d83633bf.

## Complete-source result: reject

The separately frozen eight-movie graph test used the patched official scorer
and replayed the parent, FOCUS-flow and previous constant-variance conditional
control exactly. All candidate graphs were persisted before sourceGT scoring.

| Complete-source measurement | Constant uncertainty | Conditional uncertainty |
| --- | ---: | ---: |
| Combined score | 0.7838611993 | 0.7761649944 |
| Raw edge Jaccard | 0.8012639505 | 0.7937659546 |
| True divisions | 5 | 4 |
| False divisions | 178 | 169 |
| Node recall | 0.9643973729 | 0.9643973729 |

All8 movies have worse adjusted edge scores than the constant-uncertainty
control. The original parent-relative per-movie guard also fails on67ebd073
(-0.044650, allowed loss0.02). All four final source conditions fail. Do not
promote, submit, retune on source, or extend this failed configuration unchanged.
The constant-uncertainty comparator itself was not previously promoted.

Source scoring94.125CPU seconds. Host verification replayed all168,586 unchanged
raw nodes and139,654 exact candidate edges, eight complete100frame movies,
frozen models/controls and official aggregate arithmetic. Its scope is graph
transport/prediction and arithmetic, not an additional GT-matching pass.
The source execution performed the fresh official GT matching for all four arms.
Twenty-three related tests pass1.89seconds, including analytic gradient,
portable parameters, actual ranking changes, unchanged raw nodes, empty-frame
handling, degree constraints and rejection rather than node truncation.

Source result SHA256:
ded1f7b46eb37624ec58b274dafc83ddba024ae794145a531a885375aeb7aa8f.
Source verification SHA256:
5331e041add0a43fb56d8e7aced1eeb822cf7ad78434f3ff7eb2ee2783280637.

## Consequence for the next experiment

Better residual density did not translate into better association decisions.
The residual likelihood models only true ordinary displacements; the graph
also compares competing detections and the null option. Its unnormalized
proximity weights are not the same objective as the normalized Gaussian density.
This is evidence to prioritize directly supervised candidate ranking over
another covariance-only search, not proof that a particular ranker will work.
The previous presence-only limitation remains: changing presence cannot repair
an incorrect ordering of candidate parents.

All evaluations here are fitting-correction or exposed-source development
evidence, not independent whole-system/embryo-held-out validation. No diagnostic
movies or remaining target movies were opened, and no submission was produced.
No GPU was used; all processes are terminal. Antelume named-profile STS again
returned ExpiredToken this turn. No RSNA/shared-instance mutation. Last observed
Kaggle quota8.80h at17:18 must be refreshed before any future GPU launch.
This goal turn is progress (completed model screen plus full-source rejection),
but the objective remains unfinished and today's new qualified submissions0/5.
