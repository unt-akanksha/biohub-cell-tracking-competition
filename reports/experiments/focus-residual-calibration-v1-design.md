# Calibrate motion residual uncertainty from cached training movies

Before fitting or evaluating this candidate: freeze the two available full
FOCUS/owned-flow training movies6bba_f1fde7e0 and6bba_23af9eeb. They belong to
the original training pool, not the8 source-selection movies or44b6 target
embryo. Their availability, not a new score search, determines this subset.
The small subset limits generalization confidence.

Source error attribution on all8 frozen movies found884 missed links:
271 lack an endpoint,613 have both endpoints detected. Of the613,540 true
pairs lie at/beyond the existing null-distance cost,72 are ambiguous between
parents and1 is topology-blocked. This does NOT prove that increasing a
search radius will recover them: motion/localization error and false links
remain possible. No oracle-selected edge is generated.

Estimate the mean and diagonal variance of actual source-minus-predicted-
parent residuals on the TWO TRAINING movies only. Use official node matching,
unique matched endpoints, ordinary consecutive GT edges, all such residuals
without tail trimming. Require >=50 examples per movie and >=100 pooled.
Fit maximum likelihood mean/variance, with each variance floored at native
voxel-size squared/6 (difference of two uniform voxel quantization errors).
The floor is a conservative resolution constraint, not a selected threshold.
Save fitted parameters before scoring any source candidate.

One fixed candidate: adjust only Gaussian residual mean/variance. Keep every
raw node, native flow field, null logit-4.5, posterior>.5, one parent/two
children and complete-frame scope. No radius/threshold/covariance sweep, node
count objective, new detector run or GPU use. Unlike the earlier neural/prior
coefficient calibration, this fits actual FOCUS-centroid motion residuals.

Persist all8 candidate graphs before source GT. Freshly replay parent and
FOCUS-flow controls. Apply the unchanged original source gate plus gain in
score/raw-edge Jaccard versus FOCUS-flow0.7753326, <=0.02 per-movie loss versus
that reference, identical recall and >=3 true divisions. Do not relax a failed
gate. Passing is not submission authorization or independent embryo evidence.
FOCUS pretraining overlap remains unknown; all8 source movies are already
exposed development data. Remaining target movies stay closed.
