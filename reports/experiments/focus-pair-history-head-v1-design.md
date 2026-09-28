# Previous-image-motion candidate head: frozen small optimizer test

September10. This is a new information experiment, not a rerun of the failed
presence-only or appearance-only families. The history data/functionality audit
799102689148f65daeb557a1aaf8fc002f2524fa5ccb2c2efaf527c4ee81432d passed before
this design. The concurrently completing100-tree experiment remains unchanged.

## Scientific contract

Retain the original eight real-candidate features and physical/null offsets.
Append the seven exact, uncompressed FP64 previous-image-motion features from
research/focus_pair_history.py. No new image backbone, pseudo-labels, candidate
sampling, fitted feature projection or source-specific behavior. Unknown targets
remain excluded from supervision only, never from inference or assigned negatives.

Fit the same complete-group softmax likelihood with per-group parent weight1
and absent weight sqrt(fitting_parent/fitting_absent), determined on fitting
groups only. Ridge1 on every coefficient except original real intercept.
Original current-distance quadratic corrections stay <=0.5; the three new
history-distance quadratic coefficients stay <=0. This prevents rewarding
arbitrarily large history residuals. Signed coefficients and availability
intercept are learned. All15 coefficients initialize at zero.

Use bound-preserving coordinates from the exact zero-point *fitting* Hessian,
not a change to features, objective or coefficient penalty. Six constrained
coordinates remain diagonal with positive scales. L-BFGS-B max500, gtol1e-8,
ftol1e-12; retain coordinates, every25th iterate, terminal state and model.
No tolerance/step/weight/threshold sweep follows a failure.

## Small test before any larger fit

Use exactly the original6bba_57b7cc1e/frame31 functionality packet and cached
frame30 predecessor, both SHA checked against the existing audited receipts.
Original20 known groups comprise19 parents and1 absent, with23,220 complete
source/null choices. Independently check the analytic gradient and Hessian,
zero-history-weight equivalence to the original eight-feature objective,
feasible transformed bounds, optimization loss decrease and exact model/metric
reload. Include first-frame/missing-history and label/candidate guards in tests.
Training accuracy is functionality only, not held-out model evidence.

CPU-only smoke cap180seconds; no more than2GiB supervised arrays (actual small
test is orders of magnitude smaller). No diagnostic/source/target data, GPU,
cloud operation, complete twelve-fold quality run or final submission in this
stage. Larger fitting requires a separate predeclared twelve-movie held-out
protocol with original controls and unchanged promotion gates, after the current
tree screen is terminal and verified. Passing this smoke alone is not permission
to promote or submit a model.
