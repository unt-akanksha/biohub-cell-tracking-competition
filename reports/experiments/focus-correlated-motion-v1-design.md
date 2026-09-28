# Correlated conditional motion: fixed CPU feasibility screen

Use the existing, verified 14-fitting-movie conditional-motion arrays only.
Keep the six-feature ridge mean exactly unchanged. Replace its diagonal residual
Gaussian by the maximum-likelihood full 3x3 residual second-moment matrix. Apply
the existing native-voxel variance floor as an eigenvalue floor after whitening
by that diagonal floor. This guarantees covariance >= the physical floor in
positive-semidefinite order and reproduces the diagonal rule for diagonal input.
No covariance shrinkage search, mean change, threshold change or new labels.

Compare against the previous conditional diagonal Gaussian in all 14 original
leave-one-movie-out folds. Each covariance uses the other 13 movies only. Report
proper normalized 3D Gaussian NLL and physical MSE by movie and pooled by link
count. The mean is identical, so MSE must replay exactly, not improve artificially.

Gate fixed before execution: strictly lower pooled NLL; lower NLL on at least
8/14 movies; worst-movie NLL no higher than the diagonal model's worst-movie NLL;
and exact unchanged MSE sums for every movie. No post-result gate adjustment.
Only a pass permits fitting/saving the all-14 covariance for a separate, fixed
full-source graph experiment. This screen does not authorize a submission or
claim independent whole-model validation. No GPU use or source/target access.

Rationale: diagonal uncertainty cannot describe an oblique residual error cloud.
A full covariance can adjust direction-sensitive plausibility without increasing
the marginal link radius indiscriminately. Whether this actually helps is unknown.
