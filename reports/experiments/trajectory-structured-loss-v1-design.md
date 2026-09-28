# Fixed partial-label structured assignment experiment

Motivation: independent ranking improved conditional parent counts but its actual
degree-preserving graph intervention lost one TP and added two FP. The source-only
feasibility audit bounds this family at16additional known correct choices, not a
top-five solution by itself. No oracle predictions were exported. Do not relax the
graph constraints or retune the rejected ranker's threshold on exposed errors.

Test a different learning objective on the same18prediction-only features: latent
structured hinge. For each annotated source frame, compare the highest-scoring
feasible assignment consistent with conservative known parent constraints to the
highest-scoring loss-augmented feasible assignment. Unannotated rows remain free;
ambiguous alternatives receive zero incorrect-edge margin. Conflicting division
constraints and infeasible partial constraints are omitted explicitly, not forced
into one-to-one supervision. No raw labels or oracle assignment is used at inference.

Conceptual primary source: [Yu and Joachims, Learning Structural SVMs with Latent
Variables, ICML2009](https://www.cs.cornell.edu/~cnyu/papers/icml09_latentssvm.pdf).
This is our small NumPy/SciPy partial-assignment implementation and fixed stochastic
optimization, not a reproduction of the paper's CCCP solver or evidence of Biohub
performance. No third-party SVM implementation or new weights downloaded.

Fixed training:300AdaGradupdates, batch4whole-frame problems, step size.05,
L2=.01 toward the source-only independently fitted initial weights. Uniform movie
then frame sampling, seed1729. For each held-movie fold, initialize from its exact
previous source-only fold weights, not weights fitted on that held movie. Fit both
source embryos, four held-movie folds each plus full-source fits. No hyperparameter
search, validation routing or checkpoint selection. Final update only.

Screen actual feasible, unmasked joint assignments, not independently argmaxed
parents. Require per-source pooled correct choices to strictly exceed both the
current graph and the initial ranker's joint assignments, and no held movie loss
versus current. Require analogous opposite-embryo improvements. A pass only permits
complete-movie official scoring and separate selection. Source fitting data are
not final private-test validation; public backbone training overlap remains.

Seven unit tests cover a genuinely coupled cycle, gradient finite differences,
unknown/ambiguous labels, division-constraint conflicts, and matching the frozen
assignment feasible set. Run a two-update real-data smoke before full fits. CPU
only; Antelume's existing baseline-selection collection runs independently.
