# Nonlinear full-candidate correction: frozen twelve-fold screen

The first full fitting fold has100trees and a verified native/portable replay.
Its75+25 smoke resumed EXACT original100trees. Actual-worker recovery peak
5,241,425,920B fits the unchanged6GiB cap;244.203s recovery plus170.265s stopped
prefix profile give conservative first-fold work about415s, including repeated
preparation and smoke/replay overhead. The initial guard defect and stopped
worker remain documented; no historical launcher measurement is treated as
actual GPU or CPU memory use.

Now evaluate all12 existing fitting movies, each with baseline projection,
linear coefficients and residual trees trained on exactly the other11. Keep
every existing tree hyperparameter, objective, physical baseline, full candidate
coverage and original label. Reuse the first fold's completed100tree model only
after exact artifact/baseline checks and proving its first75trees equal the
preserved native checkpoint. Do not refit that fold. All other folds use the
original tested100tree fitting implementation, releasing source arrays after
preparation and before tree fitting. No quantile implementation change.

Each fold persists its entire composite model before correction-held-out
scoring. No exposed diagnostic, source or new target movies are evaluated.
The encoder saw the fitting movies: these are correction-only screens, not
embryo-independent validation or tracking scores. No posthoc threshold, weight,
feature, candidate count, metric or gate change based on intermediate results.

Use2CPUthreads and sequential direct-interpreter workers; verify each actualPID.
Require at least7GiB free local memory before every fold. Actual worker6GiB peak
guard,900s per-fold and7200s whole-run cap. First profile model reuse is cheap;
11remaining folds at roughly400s plus checks suggests about60-90minutes, not a
guarantee. Stop on resource/numerical failure, preserve completed models and
partial25tree checkpoints, and never restart merely due to observation timeout.
No cloud/GPU/RSNA mutation. Kaggle quota must still be refreshed before any
future GPU launch, preserving8hours.

After all12, apply the EXACT original five gates: pooledNLL below physical and
original neural, correctparents>=9835, correctabsent>=115, at least8/12 NLLwins
against neural, worst per-movieNLLregression<=0.02. Compare frozen full-linear,
LDA and weighted controls without tuning or selecting per-movie winners.
Even a pass requires independent host replay, an all-fitting final model and
the existing diagnostic, complete-movie patched metric, embryo, runtime and
submission-integrity checks. Do not auto-export a final model or submit here.
