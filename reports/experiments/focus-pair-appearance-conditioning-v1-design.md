# Numerical conditioning after first-fold solver failure

The original full72D LOMO arm stopped at its fixed500iteration limit in the
first fitting fold. It evaluated no held-out movie and exported no model.
The completed LDA arm still fails its unchanged quality gate. Do not rerun
either failed configuration unchanged or increase the old iteration cap.

Diagnose the original full objective's curvature on fitting data only. Derive
the exact weighted grouped-softmax Hessian: sum of within-group feature
covariances plus the original ridge1 penalty, with unpenalized intercept.
Skipping exactly zero floating-point probability terms is arithmetic only;
all candidate rows remain in loss, gradient and prediction.

Use a coordinate transform, not a change to model/objective. Partition indices
4,5,6 (bounded squared-motion coefficients) from the unconstrained69variables.
Cholesky-whiten the free block, subtract its fitted cross-curvature with the
bounded block, and diagonally scale the bounded Schur complement. This yields
an invertible block-triangular theta=P*beta map. The original bounds remain
independent in beta; their upper values use a one-ULP inward guard against
floating-point violation of theta<=0.5. No damping or alternate solver fallback.
Ridge is still computed in original theta, not beta. Original and transformed
gradients obey the exact chain rule. The model exported later is original theta.

First test Hessian-vector finite differences, transformed gradients and bounds
on synthetic data. Then the exact original20target fitting smoke for both arms:
zero initialization, L-BFGS-Bmax500,gtol1e-8,ftol1e-12, same weights/projection.
Require objective within1e-5 of CPU control and exact every-target decisions.
Persist fitted theta/optimizer messages even on failure; never evaluate quality
from a failed fit. This is numerical functionality, not a new LDA quality run.

Only after the smoke passes, compute the first full fold's zero Hessian using
the same prepared arrays, replay every transform/label and original zero loss/
gradient norm. Persist Hessian, coordinate transform, eigenvalue condition
numbers and timing. Require at least10x condition-number reduction before
admitting a separate bounded full-fit recovery. No head optimization or held-out
scoring in the full-size curvature profile. Two CPU threads,300second cap;
no GPU/cloud/RSNA/source/diagnostic/new-target action. Quality gates unchanged.
