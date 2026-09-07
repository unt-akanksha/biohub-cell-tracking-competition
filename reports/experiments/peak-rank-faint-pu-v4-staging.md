# Peak-rank faint-cell PU v4 staging

Status: precommitted training member, queued behind the NucVerse3D
compatibility screen. It is not a submission candidate until it passes all
downstream clean-validation gates.

## Hypothesis

The v1 step-1,000 checkpoint already has strong complete-label synthetic
performance (`0.978156` mean AP) but weak sparse-real positive recall
(`0.675325`). Current public error analysis independently identifies transient
one- or two-frame fluorescence loss as a source of paired edge FP/FN errors.
The model already receives the previous, current, and following frame, so the
targeted experiment is to train it to retain a center-frame peak when local
current-frame evidence fades but temporal context remains.

## Frozen member

- Run ID:
  `synthetic256-real-conservative-pu-faint-temporal-peak-rank-v4`
- Model: project-authored 67.0M-parameter temporal ConvNeXt/U-Net,
  widths `(128,256,512,1024)`, depths `(3,3,9,3)`.
- Schedule: 2,000 steps, evaluation at 1,000 and 2,000, seed `3601079`,
  25,200-second internal wall guard.
- Complete-label synthetic loss: unchanged heatmap, peak rank, subvoxel
  offset, and auxiliary supervision.
- Sparse-real loss: positive logits, offsets, and positive auxiliary logits
  only; unannotated real voxels produce no negative rank gradient.
- Depth robustness: random axial attenuation to a minimum factor of `0.25`.
- Temporal fading: stable random training points receive a smooth Gaussian
  attenuation on the center frame, with a minority extending into one adjacent
  frame. Synthetic examples use probability `0.70`, sparse-real examples
  `0.40`, and at most 24 points are altered. Points and labels do not move.

The constants are generic augmentation values fixed before this member runs.
There is no movie ID, public prediction, leaderboard score, validation
coordinate, or competition-test access in the trainer.

## Queue and recovery contract

The member starts only after the v3 archive and NucVerse archive are locally
verified and their exact SHA-256 values are acknowledged to Antelume. Only then
does its allow-listed cleanup remove the redundant Biohub v3 result copy and
the locally recoverable NucVerse staging directory. The runner never names or
signals any unrelated workload. A 1.3 GB free-space floor and idle-A10G check
remain mandatory.

Source SHA-256 values:

- trainer: `6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c`
- runner: `fc0087517358215de851f93f111b688ea7af5b7be629527a9b5514ca60dc57de`
- GPU yield guard: `123c44a678fd94c8e8b51efb53c2a94148d4b067283cc6953a05e56f63aec340`

Eighteen focused augmentation, loss, archive, sequential-runner, harvester,
NucVerse, and coexistence-guard tests pass. Ruff is not installed in the host
environment; Python compilation, Bash syntax, and PowerShell validation pass.

The member also has distinct runtime, private dual-T4 validation-kernel, full
tracking-candidate, patched-official scorer, and one-shot submission controller
identities. The two long-lived local controllers wait on the verified v4
harvest and cannot run or submit if training, sealed audit, complete-movie
validation, runtime projection, or exact-score promotion rejects the member.
Sixteen focused candidate-path/controller tests and both controller preflights
pass.

## Promotion boundary

This archive can authorize only the existing private clean-validation stage.
It must independently pass synthetic selection, real positive selection,
sealed audit, complete-movie TTA/runtime validation, and the patched official
score gate before candidate packaging. The `0.945` target does not weaken any
gate.
