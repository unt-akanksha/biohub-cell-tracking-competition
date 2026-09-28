# Exact-GT dropout four-step functionality test

Requires completed exact-GT audit d27222f2... and full sidecar replay b8125cf3....
Use the verified1,121 fitting augmentations, selecting the largest source*target
pair and the pair with most synthetic null columns (fitting-order tie break).
Four updates: original largest, augmented largest, original most-null, augmented
most-null. Exact GT centers are training-only metadata, never inference inputs.

Original owned initializer; frozen encoder/detector/flow; head-only FP32AdamW,
learningrate1e-4,weightdecay1e-4,clip1,seed244691. Derive nullweight from combined
original/augmented fitting counts:20,387 real parents and1,443 null labels.
All original physical/neural diagnostics must replay before optimizer. No final
diagnostic score or checkpoint selection from this four-step functionality run.

Require full augmentation replay, four finite nonzero-gradient updates, unchanged
frozen tensors and exact real-packet checkpoint reload. Only passing smoke permits
a separately frozen full training run and unchanged real-data diagnostic gate.

Preserve the tested private offline two-T4 environment;900-second cap,
780-second internal worker deadline,840-second watchdog. Freshquota must leave
at least8h after0.25h declared worstcase. Sequential jobs; no cloud/RSNA changes.
Pack metadata losslessly below950,000 notebook bytes; hash the actual emitted
bundle bytes, preserving its link to original audited files.
