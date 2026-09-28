# Parent-dropout smoke passed and was host-verified

Four real fitting updates completed31.733819worker seconds; launcher85.513936s.
All1,121 augmentations replayed the full host audit before optimizer. Original
physical/neural diagnostic controls replayed exactly. Largest1160x1187 pair
tested original and augmented; a second pair tested two synthetic null labels.
Each update had finite loss/nonzero finite head gradient. Frozen tensor hash
remaineda9a07f33...; original raw inputs/diagnostic packets were unchanged.

Checkpoint step-0004.pt strict-reloaded with exact real cached-pair logits:
`11974cdfbd5462ee58d4a8885dc5cc9880f692d0d79f09a297a6e1dd87f797ea`.
Peak allocatedGPU168096768bytes (~160MiB). Fitting-only nullweight3.7307346923.
No final diagnostic or tracking-quality evaluation was performed in this smoke.

Host verifier independently checked actual runtime bytes, original audit SHA,
stress-pair selection/augmentation, controls, finite updates and checkpoint hash.
Receipt SHA-256:
`8d0a61d5c4c6e3a36f431319e1f205e50a138cdccb7ebd15d038f0ff324ba4e0`.
Worker result SHA-256:
`c72a3def73f9e5d6697f9c6432db33541d8e549bdd7eb36a8f5fa65de22e4f95`.
Follower37411/download17465/verifier68300 all terminal.

This permits a bounded full training experiment, not a submission. Do not use
the four-step checkpoint as a selected model. Proposed full800step run restarts
the original weights and alternates original/augmented fitting pairs, retaining
the original real-data diagnostic gate and fixed final checkpoint.
