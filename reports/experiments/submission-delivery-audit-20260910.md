# Submission delivery audit during motion-only validation

## Current submission history

Read-only Kaggle SDK snapshot of20 date-ordered submissions:
`recent-submission-audit-20260910.json`. Latest returned submission55784044
is dated2026-08-26 and is now COMPLETE, superseding the historical PENDING
observation in the old public-0927 output audit. Its public score is empty in
the API. Do not infer0.927 from its title or treat an empty score as zero.
No new submission was created during this audit.

The old output audit explicitly records `upstream_exact_match=true`; it is
an attributed reproduction, not the new owned candidate requested by the user.
No historical exploit notebook/output was downloaded or executed in this audit.
Public submission scores remain excluded from model selection.

## Existing two-GPU delivery machinery

`research/submission_sharding.py` passes14 regression tests. It enforces two
GPU worker ownership, disjoint whole-movie coverage, duplicate/missing-output
rejection and bounded cleanup. It reserves two hours inside the twelve-hour
notebook limit, leaving at most ten hours for inference. This runtime reserve
does not replace the separate eight-hour Kaggle quota reserve.

The actual existing model-specific entry points are
`research/trackastra_graph/dual_fold_rerank_submission.py` and
`research/temporal_contrastive/dual_fold_appearance_submission.py`.
They require their own accepted reciprocal models/configurations and are not
drop-in proof that the newer TemporalUNet/flow candidate is submission-ready.
Do not bypass their acceptance documents or relabel a new checkpoint as an old
accepted architecture. Reuse the common shard helper when a new model is
promoted, then run a real two-device end-to-end smoke for that exact inference.

Outstanding delivery evidence remains: a promoted clean model, frozen
generalization evidence, exact two-GPU end-to-end inference, offline artifact
and environment hashes, measured full-inference runtime within its declared
quota-safe cap, complete CSV coverage/schema and final submission acceptance.
The current motion-only source validation does not yet establish those items.
