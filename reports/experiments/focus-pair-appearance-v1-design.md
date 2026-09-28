# Frozen pair appearance / Fisher-LDA comparison

Prior turn was progress: weighted candidate ranking completed and failed the
missing-parent gate, and the user-supplied paper was assessed. This protocol
implements a different representation hypothesis, not another weight sweep.

Reference: [arXiv2604.03928v2](https://arxiv.org/html/2604.03928v2).
This is our pair-ranking adaptation, not a reproduction of image classification.

## Data and representation — freeze before extraction or fitting

Use only the original twelve fitting movies, all 99 adjacent transitions and
all 10,915 known target groups / 4,691,320 candidate-plus-null rows. Exact
existing group identities, labels, motion offsets and eight baseline features
must replay. Unknown targets supply no training labels. No sampled negatives,
candidate truncation, synthetic labels, detector changes or source/diagnostic/
target reads. The encoder saw these movies: LOMO tests the correction only.

Normalize each cached 32-channel source/target vector by max(L2 norm,1e-12).
For each real pair, concatenate 32 absolute channel differences and 32 channel
products. Store these 64 descriptors in float32, target-major/source-minor
order, with a zero row for the final null candidate. Cosine is the sum of the
product channels and must replay the old feature within 1e-6. Feature extraction
itself is label-free; only supervised packing selects known target columns.

For projection fitting, real pairs of a known target are positive only at its
verified parent index; all other real pairs are negative. A known-absent target
has all-negative real pairs. Exclude null rows from appearance statistics.
Use every real pair once without class/candidate sampling or row reweighting.
Persist separate class counts, sums and second moments for each fitting movie.
No global fitted projection is created before the screening gate passes.

## Two predeclared arms

Full arm: original eight features plus all 64 standardized pair descriptors.
LDA arm: original eight features plus one standardized Fisher discriminant.
Null extra features remain zero after transforms, not a standardized fake cell.
Both use identical known groups, physical offsets, weighted grouped-softmax
loss and optimizer. Retain motion/cosine rather than replacing the entire input.

Compute descriptor means and population standard deviations from real pairs
in the training fold only; std below1e-8 becomes1. Compute pooled within-class
covariance using (sum of centered class scatter)/(N-2), in standardized space.
Use the symmetric eigendecomposition pseudoinverse, retaining eigenvalues
greater than1e-8 times the largest. Reject materially negative covariance
eigenvalues or a zero discriminant; do not silently change the solver.
The direction is inv(Sw)*(positive_mean-negative_mean). Center at the training
global mean and normalize its projected population variance to1. Record rank,
cutoff, moments/counts and all scaler/projection parameters. This stable Fisher
implementation is not a claim of bitwise equivalence to the paper's SVD solver.

Head fitting: parent-group weight1, absent-group weight sqrt(fitting parent
groups / fitting absent groups), computed from that fold only. Ridge1 on every
slope, unpenalized intercept; zero initialization; squared-motion correction
upper bound0.5. L-BFGS-B max500,gtol1e-8,ftol1e-12. No threshold, weight, feature,
dimension or hyperparameter search after seeing validation results.

## Functionality, resource admission and evaluation

First synthetic checks: descriptor/cosine identity, unknown/null handling,
streamed-vs-direct moments, singular-covariance handling, fold roles,
serialization, generalized-head analytic gradient and baseline reduction.
Then the largest real fitting packet containing both target-label classes,
selected by real candidate rows and lexical stem/frame ties. Fit scaler/LDA
and both heads on that packet only; verify gradients and exact saved predictions.
This is functionality, not held-out quality.

Descriptor files are memory-mapped; bounded feature blocks avoid a full
float64 4.7-million-by72 allocation. Persist exact row coverage, packet/raw/
manifest hashes and per-movie moment statistics. Disk descriptor cap2GiB;
bounded in-memory active arrays cap2GiB. CPU numeric threads2. Initial complete
extraction plus real-smoke watchdog900seconds. Use measured smoke and complete-
data objective throughput to declare the separate full-run cap before launch.
No GPU or cloud/shared-RSNA mutations.

Subsequent full evaluation: twelve movie-held-out fits per arm. Fit every
scaler, projection and head only on the other eleven movies, persist before
held-out scoring. Replay physical/original neural and previous weighted-ranker
controls. Each arm must satisfy the unchanged prior gate: pooled NLL below
physical and neural, correct parent>=9835, correct absent>=115, at least8/12
NLL wins vs neural, and no movie NLL regression>0.02. Do not call projection
helpful merely because richer inputs beat the old cosine-only representation;
compare full and LDA arms directly. Only passing arms may receive an all-fit
final model and proceed through the existing diagnostic, full-movie patched
metric, embryo, runtime and offline-submission gates. This protocol by itself
does not authorize target access or submission.
