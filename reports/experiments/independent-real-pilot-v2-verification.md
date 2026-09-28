# Independent real pilot v2: verified, not promotable

Downloaded source/runtime files match the launch notebook, and the terminal,
split, histories, finite checkpoint tensors, optimizer state and RNG payloads
pass `scripts/verify-independent-real-pilot.py`.

- Checkpoint SHA-256: `30da7d5e348f56fea206984007e2188512a830fd30a0021182bc5c70aa5aa44d`.
- Notebook SHA-256: `789012a78bf83a543552a87f8e5096ec3a803d19c89ae6279d28873ad53fc6eb`.
- 50 detector-only warm-up steps, followed by 100 attempted joint steps.
- Joint optimizer states record between 2 and 77 updates, not 100 successful updates.
- First/last joint-block detector losses: 3.8354818 / 3.3345067.
- Five of ten blocks have nonzero association loss; the last block has zero.
- Last block has at most one detection per sample. Recall is not measured.
- Training-side elapsed time: 67.87 seconds through the final block.
- Peak allocated GPU memory: 2,571,227,648 and 2,398,110,208 bytes.

The pilot establishes functional training and checkpoint transport, not
adequate detection coverage or generalization. No longer run or submission
is authorized by this report. Investigate skipped updates and complete
checkpoint-reload/inference/scoring smoke checks before scaling.
