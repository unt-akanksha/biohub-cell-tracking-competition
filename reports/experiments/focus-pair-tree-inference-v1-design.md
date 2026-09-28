# Native nonlinear inference: complete original-frame functionality and time

The nonlinear12fold quality experiment is running and remains unchanged. This
separate functionality test uses only the original20-group CPU smoke models,
not a selected fold or a promoted candidate. No quality selection occurs here.

Use the original frozen57b7cc1e frame31 packet with1160source/1187target nodes.
Drop labels before inference. Score every original target against every source
plus null, including all1167originally unknown targets. Native XGBoost3.4.1 CPU
prediction must load the exact portable100-tree structure, keep two threads,
and add its raw residual to the unchanged full linear baseline exactly once.
No new fitting, projection change, node/coordinate modification or pruning.
Subtract the common null residual only as a group-logit shift preserving all
probabilities, keeping the external null reference at-4.5.

Compare32target native blocks against17target portable blocks on every target's
decision, original global node identity and selected/null posterior. Replay
the original20 known decisions from the previously saved fitting smoke scores.
Require exact serialized outputs and unchanged input arrays. Measure actual
native and portable elapsed time on this packet. The existing conservative
block estimate must stay below1GiB; it is not a measured process-RSS claim.

300second CPU-only wall-clock cap. No GPU/cloud/RSNA actions. Native and portable
implementations use only the isolated pair-tree environment, without changing
the active experiment's files or dependencies. Tests already cover complete
unknown-inclusive inference, block boundaries, empty frames, duplicate IDs and
wrong baseline dimensions. This does not establish hidden-set12hour feasibility
or authorize deployment of any model that failed quality validation.
