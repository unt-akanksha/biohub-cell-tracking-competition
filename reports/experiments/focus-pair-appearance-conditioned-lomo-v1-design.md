# Full-feature numerical recovery, same scientific experiment

The original full-feature first fold is terminal after500iterations, with no
held-out score and no fitted model. The admitted curvature profile finds initial
Hessian condition30274.668, reduced to1.84974 by bound-preserving coordinates.
The two original real CPU smoke objectives agree within3.1e-10, with exact
target decisions. This is numerical evidence, not a new candidate-quality test.

Recover only the uncompleted full72D arm with the frozen coordinate transform
algorithm. Do not rerun failed LDA. Every fold uses its own11 training movies
to compute the original projection/weights and zero-point Hessian. Retain zero
initialization, L-BFGS-Bmax500,gtol1e-8,ftol1e-12, identical original-theta ridge,
physical bounds, complete candidate groups and known labels. No damping or
hyperparameter retries. Validate the actual full-fold Hessian-vector finite
difference to relative error<=2e-6 before its optimizer starts. Persist theta,
beta and the exact coordinate matrix, including25iteration checkpoints and
optimizer terminal state even on failure. Model files store original theta.

The first fold can reuse the previous prepared arrays only after exact complete
transform/label replay. Others prepare bounded readonly maps as before. Two
CPU threads,9000second whole-run cap; no GPU/cloud/shared-project changes.
Expected solver cost remains uncertain until the first full fitting succeeds.
Stop the arm on a numerical failure, without evaluating that fold or changing
its settings. Preserve all partial optimization checkpoints for diagnosis.

Store each fold-only model before its held-out scoring. Apply the original
five quality gates separately, unchanged, only after12folds. Compare the
verified LDA and weighted controls but do not tune against them. Any passing
arm requires host verification before all-fit export and subsequent diagnostic,
complete-movie, embryo and offline runtime gates. No source/diagnostic/new
target access or submission in this recovery. Encoder-held-out claims remain
prohibited: the encoder used these training movies.
