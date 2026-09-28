# Whole-movie routing diagnostic: limited source headroom

On the60frozen optimization movies, baseline scores0.918039752998 and the
submitted fixed8anchor0.931693079463. A perfect, ground-truth-informed WHOLE
MOVIE chooser between those two complete valid graphs has maximum official
pooled score0.932046644453: only+0.000353564990over always using the anchor.

This is an unattainable-inference diagnostic bound, NOT a proposed GT-based
router, candidate result, new submission score, or leaderboard forecast. No
movie-level oracle decisions, features or graphs are exported. Local component
routing and new models are NOT bounded by this two-whole-graph comparison.

The bound uses the pinned official scorer and prior hash-verified source rows.
For each movie, nodes, adjustment ratio, GTedge count and division counts are
equal across arms. Therefore the variable pooled score is a ratio:
sum(adjustment_factor * edgeTP) / sum(GTedges + edgeFP), plus the fixed division
term0.009722222222. Fractional optimization converged in3iterations with
certificate residual-6.54e-13. It reproduces both original arm scores to1e-12.

Simply choosing each movie's larger individual score is NOT guaranteed to
maximize the pooled score because denominators/weights also change. The helper
is tested against exhaustive enumeration for20six-movie cases and an explicit
counterexample; invalid inputs rejected. Five tests pass,1.07seconds.

Conclusion: avoiding a few whole-movie regressions is valid but has little
headroom on this source set. Prioritize genuine local/event-model improvements;
do not spend expensive GPU time fitting a complex movie-identity-like gate to
try to recover a source gain whose perfect-choice ceiling is only0.00035.

Report trajectory-movie-routing-bound-v1.json; no new GT files opened. All
source scores used were already exposed in earlier source experiments.
