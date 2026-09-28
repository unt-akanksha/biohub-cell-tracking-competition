# Same four-step joint smoke with bounded AMP scale recovery

The prior smoke replayed all4 original image/features/logits, completed one
encoder/head update, then stopped before step2 optimizer update because the
largest pair had nonfinite gradients. Finite forward loss was checked before
backward. This is consistent with loss-scaling overflow, not proof of its cause.

Retain original checkpoint, identical four stresspairs, FP32 replay, model,
normalization, learning rates, clipping, seed, weighting, frozen detector/flow
and all previous quality/nonfinite guards. InitialGradScaler scale65536 stays
unchanged. Only add bounded same-example scale recovery, at most17attempts per
successful update. On nonfinite unscaled gradients, letGradScaler skip optimizer
and reduce scale. Assert parameters unchanged. Restore pre-attemptCPU/CUDARNG
before retrying so dropout draws stay identical. A nonfinite forward loss,
finite-gradient norm failure or exhausted retry budget still aborts.

Require4actual finite updates, both encoder/head gradients and changed tensors,
unchanged detector/flow, exact finalreal-image checkpoint replay. Log every
skipped attempt/scale and peak memory. No fitting/threshold/model choices or
diagnostic/source/target evaluation. Passing only authorizes larger-run planning.
PrivateofflineGPU job cap1h; recheckquota and preserve8h. No sharedproject changes.

The skip/backoff ordering follows the official PyTorch AMP examples: unscale
before clipping; GradScaler.step skips updates with inf/NaN gradients, and
update adjusts scaling. https://docs.pytorch.org/docs/2.14/notes/amp_examples.html
Runtime remains the pinned Kaggle environment; no PyTorch upgrade is performed.
