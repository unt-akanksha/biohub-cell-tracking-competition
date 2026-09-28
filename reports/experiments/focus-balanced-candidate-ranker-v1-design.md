# Rare-parent-absence weighted full-candidate ranking

Freeze before any fit. The unweighted eight-feature candidate ranker improves
parent ranking but fails absent-parent protection (28/161 versus 115 required).
This distinct experiment tests whether fully optimized candidate supervision
with a training-count-based rare-class weight can retain ranking and abstention.
It is not another run of the failed neural head, nor a threshold sweep.

Reuse the verified 12 fitting movies and all 4,691,320 candidate/null rows.
No negative sampling, candidate truncation, synthetic labels, new image features,
source/diagnostic/target access, public checkpoint adoption, or GPU work.
The encoder saw these movies; leave-one-movie-out tests only the correction.

Exactly the previous eight features, physical offsets, null logit -4.5, ridge 1
on slopes, unpenalized intercept, zero initialization and squared-correction
upper bound 0.5. Change only each target group's cross-entropy weight:
parent weight 1; absent weight sqrt(number of fitting parents / fitting absences).
Compute these counts separately from the eleven training movies in each fold.
No normalization, validation-based weight search or post-fit logit adjustment.
L-BFGS-B max 500 iterations, gtol 1e-8, ftol 1e-12 as before.

Before full training: synthetic analytic-gradient and serialization tests,
then a real smoke on the largest fitting frame packet containing both label
classes, selected by real candidate-row count (lexical stem/frame tie break).
The smoke uses that packet's own fitting counts and is functionality only.
Persist parameters and verify exact predictions after reload. No quality gate
is decided from smoke values.

Then twelve leave-one-movie-out fits, saving models before held-out evaluation.
Use the exact unchanged gate from focus-candidate-ranker-v1: unweighted pooled
NLL below physical and original neural controls, correct-parent count at least
9,835, correct-absent count at least 115, at least 8/12 NLL wins versus neural,
and no movie NLL regression greater than 0.02. Also report the unweighted
ranker's prior held-out results, without weakening any acceptance condition.
Only a pass authorizes an all-fitting final fit; failures cannot open the
diagnostic/source/target pools or produce a submission.

Record conditional ranking ceiling, parent rejections and wrong-parent choices
for error attribution only, not choosing weights or thresholds in this run.
CPU: two numeric threads, existing 2 GiB packed-array bound, separate 300-second
smoke and 900-second full-run watchdogs. Pin actual data, prior experiment,
code, design and saved model bytes; replay held-out predictions, objective
values, class weights and gate arithmetic. No claim of optimizer global
optimality or fresh whole-system held-out validation.
