# Complete training-only FOCUS cache and sparse-label audit

September 10, 2026. Kaggle `biohub-focus-adaptation-cache-v1/1` completed
in 2130.651 seconds (35.51 minutes). All 800 new training frames and six
previous replay frames verified. New movies contain 174,818 raw nodes;
zero failed frames, no postprocessing, exact replay, no annotations opened
during detection. All artifact, runtime and notebook identities passed.

Subsequent CPU-only label audit completed in 10.891 seconds. Fitting:
2,818 known-parent and 52 known-absent targets. Diagnostic: 2,645 known-parent
and 27 known-absent targets. Unknown targets remain ignored (137,741 fitting,
29,643 diagnostic), not negative examples. Four/four roles are fixed before
looking at labels. Diagnostic movies were included in original encoder/head
training, so they are not independent validation.

Evidence is in `focus-adaptation-cache-v1-result.json` and
`focus-adaptation-labels-v1-result.json`. This is an input/label result,
not a tracking-score improvement or promoted submission.
