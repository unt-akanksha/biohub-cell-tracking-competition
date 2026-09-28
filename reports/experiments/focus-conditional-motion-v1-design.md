# Conditional motion correction: training-movie-held-out feasibility

One fixed six-feature ridge model predicts source-minus-flow-predicted-parent
residualZYXum from target backward-flowZYXum and raw target positionZYX. Fit
intercept plus standardized features, ridge1 on slopes only, using the14
already fixed fitting movies. Fit a single diagonal Gaussian residual variance
after mean correction, retaining the native-voxel variance floor. No interactions,
polynomials, coefficient sweeps, thresholds or source-selected hyperparameters.

Recover exact ordinary unique matched adjacent links using the prior14movie
protocol. Verify residual arrays reproduce the previous extraction statistics.
Persist all features, residuals and matched pair identities before any fit.
Divisions, absent parents, ambiguous matches and unknown detections do not
enter this residual-only model. No target/source labels are opened here.

Evaluate14 leave-one-movie-out folds. Each fold standardizes, fits ridge and
fits uncertainty using only the other13 movies; the paired baseline fits its
globalGaussian using those same13 movies. Report proper GaussianNLL including
normalization and squared3Dphysical residual error, pooled by link counts and
per movie. Gate: lower pooledNLL andMSE, lowerNLL in at least8/14movies, and no
increase in worst-movieMSE. No model/fold/weight choice from results.

Only if this gate passes, fit once on all14 and save the portable model. A pass
does not establish tracking improvement or authorize source expansion/submission.
A separate fixed inference experiment would retain null/topology/rawnodes and
require the unchanged full-source gates. Component-level held-out calibration
is not independent validation of pretrained flow/detector, which saw training
movies or have unresolved pretraining overlap. GPU use0; shared projects untouched.
