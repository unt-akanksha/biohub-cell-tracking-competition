# FOCUS-specific association-head adaptation v1

Prepared September 10, 2026 before optimization. No new detector, ensemble,
metric or topology changes. Public author FOCUS proposals plus our owned
encoder/flow and adapted association head; not an exact public pipeline.

Use only the verified eight-movie frozen feature cache. Four fitting movies,
four diagnostic movies, full99-pair coverage. All roles/labels frozen before
training; unknown targets are ignored. Diagnostic movies are excluded from
this adaptation, but were included in original encoder/head training. This
diagnostic screen cannot establish independent quality or public superiority.

Initial owned checkpoint76f7da6e..., model tensors41e82ccf.... Freeze the image
encoder, detector and embedded flow. Optimize only the existing association
transformer, on cached exact FOCUS-node features and positions. Use the prior
training-only residual Gaussian (fitSHAfcef805e..., no new fitting), weight1
plus neural logits weight1; existing sparse parent/null objective, null-4.5.

One fixed run:800 updates, AdamW lr1e-4, weight decay1e-4, clip norm1,
seed244691, shuffled cyclic fitting pairs with known supervision and at least
one source. No hyperparameter sweep or leaderboard feedback. FP32 training
on GPU0. No independent work is launched on GPU1 merely to fill allocation.
All diagnostic examples remain outside the optimizer path.

Small real smoke gate after four updates, counted in the800: finite nonzero
head gradients/loss, unchanged frozen tensors, checkpoint written and
restricted reloaded, identical eval logits on the same real pair. Abort if
any check fails; persist the smoke checkpoint if successful. Final checkpoint
and optimizer/random states are recoverable; no source/target access.

Evaluate full diagnostic supervision at step0 and step800 only, with the same
physical-only control. Aggregate NLL by supervised target count, and separately
count correct known-parent and known-absent classifications. Feasibility passes
only if final NLL is strictly below BOTH controls and neither correct-parent
nor correct-null counts is below either control. No posthoc gate relaxation.
This is not the official tracking metric or a submission promotion.

Only a passing result may proceed to complete-movie tracking evaluation on
the already exposed fixed-eight source set under the unchanged existing gates.
The remaining65 target movies stay closed. No automatic submission.

One-hour declared Kaggle cap, inherited3480-second worker watchdog; sequential
launch only after fresh quota leaves at least8h beyond declared runtime.
Expected head-only compute is minutes, but not benchmarked yet. Private,
offline, exact-version input artifacts, no TPU. AWS/RSNA untouched.
