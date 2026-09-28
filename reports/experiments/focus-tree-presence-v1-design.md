# Nonlinear presence decision model v1

Separate intervention after linear presence fitting failed missing-parent
correctness. Raw features are preserved: fitting-only separation audit found
79.67% raw-cosine pairwise wins, versus73.53% after centering; this does not
prove feature collapse or justify centering/retraining the encoder.

Use exactly the existing verified frozen summary inputs. Four fitting movies
only,2,818 known-present and52 known-absent labels. No unknown negatives.
Fit a standalone binary GradientBoostingClassifier, not a correction initialized
from the previous fitted coefficients. Eight inputs: the seven existing context
features plus original presence log-odds. Default fitting class-prior initializer.
Fixed100 trees, depth2, learning rate0.05, min leaf20, subsample1, seed244691,
log loss, no early stopping and no class weighting or parameter sweep. All
other installed sklearn1.9.0 defaults recorded in the resulting JSON.

Source: [official GradientBoostingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
Local installed version/signature independently inspected; live docs are1.9.1.
The CPU worker receives only a hash-bound fitting file. Export tree arrays,
thresholds, values and initial log-odds to JSON. Float32-converted inputs and
double threshold comparisons must reproduce native fitting logits within
absolute1e-12, and JSON reload exactly. Small synthetic test before real fit.
Persist the fitted JSON before any corrected diagnostic prediction.

Inference combines learned presence with the unchanged original conditional
real-parent distribution, selecting the largest joint class. Original fixed
node inventory, parent ranking, detector/encoder/flow remain unchanged.
Same feasibility gate: diagnostic NLL below original neural and physical,
parent count at least2585, known-absent count at least18. No rule changes if
it fails. Success would still require complete-movie tracking validation and
the existing original promotion gates. Diagnostic movies were part of the
original encoder training, not independent validation. No GPU, source/target
access, automatic extension, submission or AWS/RSNA mutation.
