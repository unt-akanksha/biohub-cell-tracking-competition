# Morphology-only division screen, frozen before v7 feature/score extraction

This small CPU experiment tests low-dimensional nucleus-profile ratios rather
than another high-capacity image encoder on the same sparse division labels.
It does not modify current event fits, the submitted head or the failed router.

The primary study [Gilad et al., symmetry-based mitosis detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6662301/)
motivates daughter similarity and temporal intensity differences in fluorescence
tracking. It addresses symmetric divisions and is not evidence that these cues
work for all Biohub events. The newer public notebook's nucleus-size EDA is a
hypothesis, not a model score. Normalized nuclear fluorescence crops are NOT
calibrated cell-volume or dry-mass measurements; do not impose exact biological
mass conservation or claim that every division yields equal daughters.

Freeze existing v3+v5image-derived triplets/roles/labels unchanged:3,108rows,
65optimization positives/2,305negatives and7selection positives/731negatives.
These selection movies were exposed to earlier experiments; this is not a new
independent cohort or complete-movie test. Unknown pairs remain excluded, not
newly labeled negative. No full-movie event feature join from sparse packets.

Features from finest0.8125um crops: local shell-subtracted intensity mass proxy,
central amplitude, soft-volume proxy, weighted RMS radius, covariance elongation
and surrounding-signal ratio. Central radius3.25um; background shell4.0625to
5.6875um. For each descriptor, fixed daughter/parent log ratio and daughter
asymmetry; mass/volume use daughter sums, other descriptors use means. Add the
same6physical geometry features. All18features are daughter-order invariant;
no movie/embryo IDs, truth positions, known node counts or classifier confidence
thresholds enter the descriptor.

Fit three paired source-only linear logistic classifiers per embryo:
geometry6, geometry+legacy crop statistics30, geometry+new morphology18.
Same source-only normalization/std floor0.1,clip10,class-balanced logistic loss,
L2weight coefficient0.01 and deterministic L-BFGS convergence(max2000iterations).
No hyperparameter/seed/threshold sweep and no source epoch selection.

Calibrate each model with nextafter(max source-selection negative probability,
+infinity), the strict max-negative rule of the v3classifier (float64 here).
Do not relax a threshold if it rejects all positives. New morphology must have
zero source FP,at least50%positive recall,no fewer TP than either paired control,
and lower balanced NLL than both controls. Only then open that fixed model and
its controls on opposite-embryo selection; apply the same requirements, using
unchanged source thresholds. No ensemble/complete-movie experiment/submission
is automatically authorized. Results must report per-movie counts and the tiny
positive sample. Original geometry/statistics controls are refitted with this
same CPU optimizer; they are not claimed byte-identical to earlier CUDA fits.

First pass synthetic symmetry/gain/shape/reload tests and a two-packet
optimization-only feature smoke before full extraction or fitting. Use local
cached crops and CPU only; no Antelume or Kaggle GPU required for this screen.
