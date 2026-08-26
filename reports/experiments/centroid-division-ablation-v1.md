# centroid-division-ablation-v1

Status: staged

## Hypothesis

The clean 0.927 notebook's test path refines all predicted centroids before graph
repair, but its held-out validator omitted that refinement. A complete-movie 2x2
ablation will determine whether centroid refinement and safe-division repair each
improve the clean proxy score, interact positively, or should be removed.

## Design

- Parent: `public-0927-clean-repro-v2`
- Four complete, division-aware held-out train movies across both embryo prefixes
- Test-stem overlap excluded before held-out selection
- Frozen dual-seed TemporalUNet3D predictions and checkpoints
- Arms: centroid on/off crossed with safe-division on/off
- Official-formula adjusted edge Jaccard plus division Jaccard proxy
- No leaderboard submission from this calibration run
- 1.00 hour declared GPU budget; watchdog hard stop at 50 minutes

## Promotion gate

Choose a post-processing arm only if it improves the pooled clean proxy without
a material worst-movie or division regression. Public leaderboard score is not a
selection signal for this experiment.
