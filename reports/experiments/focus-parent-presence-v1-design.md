# Training-only parent-presence correction v1

Hypothesis: a separate context-dependent no-parent probability can improve
missing-parent recognition without changing the original owned parent ranking.
This is a new intervention, not relaxation of the failed head-adaptation gate.
Use the original checkpoint76f7da6e..., not the rejected adapted checkpoint.

Use the same verified four fitting/four diagnostic FOCUS feature movies.
Unknown labels remain excluded, not negatives. No source or target access.
Diagnostic movies were used in original encoder/head training, so this screen
is not independent validation. No leaderboard-based selection or metric hack.

GPU summary collection first: frozen original neural head plus previous
training-only calibrated physical prior. Replay the complete previous physical
and initial neural diagnostics before collection (classification counts exact,
NLL absolute tolerance2e-6), then collect compact summaries of known targets.
Seven fixed context features: best physical score, physical logsumexp, best
joint score, top-two joint margin, normalized conditional entropy, nearest
physical parent's image-feature cosine, and distance between best joint and
best physical parent. No absolute location/movie ID or graph-count targets.

Factorize joint parent/null probability into binary presence and conditional
real-parent probabilities. At zero correction this must reconstruct the
original fixed-null objective exactly (within the declared float tolerance).
Initial presence log-odds are logsumexp(real scores)+4.5. Fit an additive
linear correction to those log-odds using ONLY fitting labels. Standardize
features with fitting mean/std; std below1e-8 becomes1. Sum binary NLL plus
0.5 squared coefficient norm, excluding intercept; fixed ridge1. L-BFGS-B,
zero initialization, max500 iterations, gtol1e-8/ftol1e-12. No sweep, reweighting,
resampling or diagnostic coefficient choice. Save the fit before evaluating
diagnostic summaries. Conditional real-parent ranking remains unchanged.

Choose the largest joint class, not aggregate real-parent probability.
Feasibility requires diagnostic NLL strictly below original neural and physical
controls, correct-parent count at least2585, and correct-absent count at least18,
the SAME existing gate. Also replay zero-correction baseline from actual summaries.
If it fails, do not extend to source tracking, alter coefficients, or submit.
If it passes, complete-source tracking and original promotion gates remain
required; binary metrics are never substituted for the official scorer.

Local tests must pass before launch. One sequential private/offline GPU summary
job, one-hour cap with fresh quota leaving8h after worst case; existing3480s
watchdog. CPU fitting afterward consumes no GPU. Raw detector/encoder features
are reused. AWS, RSNA and all other projects remain untouched.
