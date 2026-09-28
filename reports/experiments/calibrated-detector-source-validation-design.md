# Conditional calibrated-detector validation

Declared while the full training-confidence collection is running, before its
diagnostic result or any calibrated source score is available.

Only if the full96-fit/24-diagnostic calibration passes: evaluate the frozen
sparse-control checkpoint0f441c6e with its single training-fitted cutoff,
unchanged D4 and standalone flow, on the original eight complete source movies.
This is a new calibrated sparse-detector recipe, not retroactive promotion of
the rejected PU checkpoint. The prior PU gates and failures remain unchanged.

Primary reference is the strongest retained frozen D4/flow model:
score0.6298395328, baseline report SHA
db9d75ad43a9bc74d3f38f3aef48e5e617510a6abb93205a0387e726415d080f.
Require combined-score and raw edge-Jaccard gains, mean node-recall loss<=.005,
at least5/8 improved adjusted-edge movies, no movie loss>.02, and worst-movie
loss<=.01. Report TP/FP/FN, every movie, divisions and runtime. Compare with
uncalibrated sparse and rejected PU as secondary diagnostics; neither replaces
the primary frozen reference or hides a failed primary condition.

The threshold comes only from training annotation recall, never a desired node
count or source/target score. No threshold sweep, graph pruning, fabricated
nodes or altered metric. The eight repeatedly consulted source movies are
development evidence, not an unbiased private-score estimate. A source pass
still does not authorize submission or opening more target movies without a
separately frozen embryo audit.

The full calibration's diagnostic failure blocks this run completely. Preserve
that failure rather than substituting the small-probe cutoff or changing the
24-movie acceptance condition after seeing its results.
