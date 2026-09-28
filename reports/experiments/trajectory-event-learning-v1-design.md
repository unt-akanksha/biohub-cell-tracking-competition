# Source-only event learning: first real functionality fit

The submitted structured trajectory candidate remains frozen. The event model
is new research, not a promoted checkpoint or a claim of leaderboard gain.

Before examining fit outcomes, freeze a small real-data smoke: use batch 0's
first two source6 movies in its existing image-only order. Use their known
two-daughter frames and one deterministic known ordinary frame per movie.
This selection tests rare-event constraints and ordinary links, not quality.
No selection or validation movie is opened. This smoke is not the full fit.

Thirty prediction-only features consist of the existing 18 edge features
(summed over both daughters for a fork) plus event indicators, sister geometry,
masked mother/future motion and birth/death boundary distances. Unknown cells
are not background or birth labels. Existing divisions and synthetic/gap
incidence remain protected; removal of old false divisions is outside this v1.

Initialize the 18 edge weights from the frozen structured v1 source6 head.
Initialize the division bias to log((D + 0.5)/(N - D + 0.5)), using annotated
parent opportunities N and annotated divisions D from fitting movies only.
This is a conservative regularization anchor, not calibrated prevalence:
the source movie collection is division-enriched and annotations are sparse.
All other event coefficients start at zero. The old runtime-only -6 pilot
coefficient is not imported as a model choice.

Optimize the exact latent partial-label hinge using three fixed smoke epochs,
Adam learning rate 0.03, seed 20260914, gradient norm cap 5, and L2 anchoring
0.1 for original edge coefficients / 0.02 for the twelve new coefficients.
Every smoke case is visited per epoch. Save each epoch without choosing the
best source score. Solver time limit is five seconds per oracle; timeout stops
training, never creates fake ground truth or silently discards a case.
Incompatible source constraints are reported and prevent smoke acceptance.

Smoke acceptance requires finite changed weights, successful exact oracles,
reproducible saved inputs and three completed epochs. Report before/after
training hinge and known-link disagreement; these are functionality diagnostics,
not independent quality evidence. Do not export submission graphs from smoke.

Next, fit source experts and evaluate full 100-frame graphs with embryo-held-out
and per-movie controls. The broader objective is clean private-test improvement,
not simply fitting rare source forks. Promotion still requires patched official
micro-scoring, worst-movie analysis and independent offline runtime verification.
