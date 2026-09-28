# Independent base checkpoint audit

The currently running FOCUS bridge diagnostic uses an all-training secondary
model. All four evaluation movies occur in that model's training manifest.

The cached support50 primary and teacher-primary last checkpoints were read
on CPU with `weights_only=True`. Both report epoch 402 and contain model,
optimizer, configuration, method, fold and metric fields. Neither contains
training or validation movie lists. Their independence is unverified.

The official repository links the author's baseline notebook:
https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission

Its source and metadata were downloaded into
`.biohub/research/official-baseline-audit-20260909`. It attaches
`thibautgoldsborough/cellmot-baseline-artifacts`. The full dataset file listing
(page size 200, no next-page token) includes an 8,357,783-byte checkpoint and
configuration but no training split manifest. No public predictions were
downloaded or used. The current official training script supports a fallback
seed-0 split; that is not evidence of which split produced the published
checkpoint. Do not infer held-out status from that fallback alone.

No replacement checkpoint has yet been authorized as independently held out.
If original split provenance cannot be established, a clean training run with
a recorded split or a model demonstrably pretrained outside Biohub is needed
for independent generalization testing. This audit launches no GPU jobs and
does not change the running bridge diagnostic's thresholds or movies.
