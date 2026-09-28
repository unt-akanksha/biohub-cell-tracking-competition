# Fixed conditional motion: full-source graph test

Proceed only after exact replay of all14 training-held-out fits/metrics and the
saved final model508d0589... from conditional-motion-v1. No fitting occurs here.
Mean correction uses original predicted backward flow plus raw targetZYX only.
Add this predicted residual correction to target flow, retaining all original
raw coordinates. Use saved residual variance, null-4.5, posterior>.5, max2children
and1parent. No thresholds, weights, graph rules or model choices tuned on source.

Source scope exactly the existing8complete100frame movies. Verify all source
flow/raw/parent provenance. Persist all candidate graphs before sourceGT.
Replay original parent and FOCUS-flow scores with patched official scorer.
Apply unchanged original source and FOCUS-flow gates, plus strict combined
score/rawJaccard gains over the prior2movie calibration and preservation of
its3true divisions. Report all movies, pooled counts, embryo and worst-movie
comparisons. No source-specific corrections, node pruning or false divisions.

A pass is development evidence only and does not authorize submission before
independent/embryo-held-out validation and full offline inference acceptance.
No remaining target movie opened in this run. CPU only; no sharedGPU impact.
