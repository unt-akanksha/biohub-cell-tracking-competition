# Fourteen-training-movie motion calibration

Prospective single CPU experiment. Preserve the original two motion fitting
movies and add the fixed twelve fitting movies already cached for association
training. The unchanged four diagnostic movies, source eight and all target
movies are excluded from fitting. This is a data expansion, not a parameter sweep.

Verify original source/feature provenance. Recover all target-centroid backward
flow from99 ordered packets per additional fitting movie, preserving raw node
IDs/coordinates. Match training nodes using the same official7um matching rule.
Use only unique matched endpoints of ordinary consecutive annotated edges;
exclude every GT parent with outdegree other than1, including cases with only
one detected division daughter. Unknown detections never become negative labels.

Replay the original two-movie residual counts/mean/variance against the frozen
fit. Then pool their residuals and the twelve new movies, each requiring at least
50 observations, and apply the unchanged diagonal Gaussian MLE/native-voxel
variance floor. No clipping, trimming, movie-specific parameters or null sweep.
Save fit and provenance before predicting/scoring source movies.

Inference changes only fitted mean/variance. Keep all raw nodes, owned flow,
null logit-4.5, posterior>.5, max2children/max1parent and complete100frames.
Save all8 candidate arrays before source GT. Fresh controls must exactly replay.
Apply original source and FOCUS-flow gates unchanged. Additionally require
strict score/raw-edge improvement over the previous two-movie calibrated arm
and preserve its true divisions. Report all movies and worst-movie outcomes.

A pass permits further validation only. Eight source movies are exposed
development, not independent confirmation. No target expansion, submission,
GPU use or modification of RSNA/shared environments in this experiment.
