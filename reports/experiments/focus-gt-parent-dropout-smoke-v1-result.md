# Exact-GT missing-parent augmentation: functionality passed

Kaggle `indarkarhana/biohub-focus-gt-parent-dropout-smoke-v1/1` completed.
Host verifier completed successfully before full training was staged.

Four stress updates, all 1,121 augmentation replays, unchanged original control
diagnostics, frozen component hashes and exact real-logit checkpoint reload
passed. Eight current smoke/augmentation/training tests passed in 1.84 seconds.
Worker 29.148 seconds; launcher 73.036 seconds; 900-second declared cap.
Peak allocation 168,129,536 bytes.

Worker result SHA256:
bd0ee23ae85c79d7142c78277e6b6aceeac71cc1bf6cc2ab31cde3899978d9e1.
Checkpoint SHA256:
e0a43eb0d4c664bdb74d20eb81a65119ecc30470be74aa28fe2266f644f5b93e.

This establishes functionality only, not quality or submission readiness.
The separately frozen 800-step training run starts from the original owned
checkpoint, not this smoke checkpoint. It requires unchanged real-label gates
before any complete-movie source evaluation.
