# Fixed conditional motion uncertainty screen

Freeze before execution. The previous motion ridge model improved14movieLOMO
NLL/MSE but failed full-source promotion; global correlated covariance also
failed. New hypothesis: uncertainty varies with target position and backward
flow, so one diagonal covariance for every target discards useful ranking
information. Keep the existing conditional mean exactly fixed. Fit only a
six-feature log-linear diagonal variance model using the same verified14
training-movie arrays and unchanged physical native-voxel variance floor.

Features: existing backwardZYXum and targetnativeZYX. Use the mean model's
training-only center/scale. For each axis:
variance = native_variance_floor * (1 + exp(intercept + standardized_x @ slope)).
Fit ordinary proper Gaussian NLL with ridge1 on slopes, unpenalized intercept,
L-BFGS-B max1000, fixed tolerances1e-8 gradient and1e-12 objective. No feature,
regularization or threshold search. Fit variances to training residuals of the
fold-specific mean model; do not fit means again using held-out observations.

Leave one training movie out at a time. Replay the actual previous conditional
mean/constant-diagonal control on all14folds. Require lower pooled proper NLL,
at least8/14 per-movie NLL gains, no increase in worst-movie NLL, and exactly
unchanged mean/MSE. Failure stops without a final model or source evaluation.
A pass permits fitting/persisting the final model on all14fitting movies and
a separately frozen full-source comparison with unchanged promotion guards.

No four-movie diagnostic or target reads. These folds test the correction only;
the underlying representation is not independently embryo held out. CPU-only,
300second cap. Archive all fold coefficients, hashes, metrics and finalmodel
if eligible. No changes to original completed artifacts. At this stage do not
alter graph probabilities/null/degree rules or create a submission.
