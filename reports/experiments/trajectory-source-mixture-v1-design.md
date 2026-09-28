# Preserve source-specific motion regimes: fixed two-expert mixture

The pooled AR2 refit is closed as neutral: none of its16 eligible endpoint
pairs pass, despite cross-source frozen experts recovering2TP without addedFP.
It is not a deployment candidate. This motivates preserving the two separately
fitted motion regimes rather than replacing them by a single covariance.

This is a NEW deployment hypothesis, selected after the initial diagnostic and
pooled failure, not a predeclared winner or untouched validation. Do not edit or
relabel either earlier experiment. Freeze this policy before computing mixture
outputs or opening the four additional division-positive movies for this policy.

Keep both already frozen source AR2 experts and each original source threshold.
For every movie, including unknown embryo IDs, retain a proposed endpoint link
if either expert supports it (minimum normalized squared motion error<=1).
All existing raw reciprocal-probability>=.88, two-sided unbranched history,
genuine detector ID, adjacent-time and empty-endpoint constraints remain.
No source weighting, recalibration, new fit, metric-derived threshold, movie ID
lookup, weak-edge deletion, node change, or division construction. Model scores
are positive-trajectory compatibility, NOT correctness probabilities.

This is a two-component motion-support mixture, not evidence that both experts
individually improve a hidden leaderboard. The neural detector/linker backbone
is still the pinned public original. The mixture's extra false-link risk must
be checked explicitly. If four diagnostic graphs equal the saved successful
cross-source graphs, verify equality and reuse that exact official score rather
than reopening labels. Otherwise apply unchanged full-movie gates. Then verify
the four additional division-positive complete movies listed in
trajectory-endpoint-deployment-v1-design.md, with unchanged joint8movie raw/
combined improvement, every movie/embryo nonregression and division counts.

No threshold search or alternate committee rule after this recipe fails.
Further offline dual-worker runtime, licensing and reproducibility checks remain.
Public backbone training overlap and historical movie exposure are explicit;
do not present the diagnostic as pristine model-level validation or0.945+LB.
