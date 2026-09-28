# Expand fitting data without changing diagnostic membership

The first four fitting movies provide only52 known-absent examples. Fixed
head adaptation, linear presence and shallow boosted presence all fail the
same diagnostic absence requirement. This motivates a data-coverage test,
not another sweep of those models on the same fitting set. More data is a
hypothesis, not a guarantee of improvement.

Select the next eight movies from the original fixed96 fitting ordering,
excluding the four cached fitting movies and two replay movies. Membership
is fixed before labels, scores or density are examined. Preserve the exact
four diagnostic movies; none becomes fitting data. No source-selection or
target movies. Original model pretraining caveats remain unchanged.

Reuse the exact completed adaptation detector notebook, changing only movie
scope, run identifiers and explanatory metadata. Exact author FOCUS weights,
audited runtime and inference unchanged. Reproduce six earlier detector smoke
frames; save every raw centroid and complete per-movie artifact. No labels,
postprocessing, node pruning, graph creation, model fit or submission.

One sequential private/offline two-T4 job, no TPU. Expected35-50 minutes based
on the prior35.51-minute eight-movie run, declared hard maximum3600 seconds
with the existing watchdog/checkpoint behavior. Fresh quota must leave8h
after that full maximum; reject if less than9h remain. Small scope/builder
tests and actual prior cache verification are required before launch.

After completion, verify every raw artifact and exact replay before opening
new fitting labels. Inventory known-parent/known-absent coverage with the
same conservative label policy. No training or full extension is automatic.
Existing failed models remain rejected; any expanded-data model needs a new
frozen design and the unchanged diagnostics/full-movie promotion requirements.
Do not repeat the FOCUS detector simply to change a downstream classifier.
