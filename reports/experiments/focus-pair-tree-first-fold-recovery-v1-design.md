# Recover the stopped first fitting fold without repeating its75trees

Original profile87429 is terminal. At20:18UTC actual workerPID30456 consumed
6,676,160,512 working-set bytes, peak6,563,136KiB, above the declared6GiB cap.
The original memory guard incorrectly measured Windows venv redirectorPID2372
(about4.5MB). Read-only process ancestry/command identity established30456 as
that profile's numerical child, and only30456 was stopped. No other project or
AWS/RSNA process was touched. Original failure receiptc68365d9... preserves the
incorrect launcher memory measurement; it must never be cited as actual cost.
Three native checkpoints survive, including75treesSHA256
cf98863fc4ac97338a8a10349ad28f27482ce5caada877fbf5f1e51ecf4a7241.

Repair execution only. Invoke the actual base interpreter directly with-S and
only the isolated pair-tree environment's site-packages explicitly on sys.path.
Before numerical recovery, a64MiB allocation child must prove that memory is
observed and the owned child can be stopped at32MiB. The numerical worker writes
itsPID before data loading, and it must exactly match the monitoredPopenPID.
Keep900second and6GiB real-worker limits. Do not raise the budget.

First resume the original20-group real smoke's75tree checkpoint for25remaining
rounds. Require EXACT original100trees and scores, not merely similar loss.
Before full recovery, replay every preserved feature, baseline margin and group
against the original11 fitting movies, then release those redundant source
maps/base arrays before constructing the SAME DMatrix. This frees roughly1.5GB
without changing the candidate matrix, quantile algorithm, objective, weights,
initial baseline, tree settings or candidate coverage. No QuantileDMatrix or
hyperparameter change. Preserve failed artifacts; new outputs live separately.

Resume the preserved75tree full-fold checkpoint for exactly25more rounds.
Export and replay native/portable/reloaded100tree predictions. Label training
traces honestly: only rounds76-100 are recorded by the resumed objective;
original earlier per-iteration losses were not persisted. No padded model is
deployed; zero-leaf padding is only an internal numerical prefix replay through
the existing100tree evaluator. No held-out, diagnostic, source or new target
array access/scoring. No quality or submission claim from fitting improvement.
