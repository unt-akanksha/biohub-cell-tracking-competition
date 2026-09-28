# Four-step candidate-dropout training smoke

Requires the completed full fitting-data audit dd75e564... before staging.
Start the original owned integrated checkpoint, not a failed adaptation model.
Freeze encoder/detector/flow and train only the association head, FP32, original
AdamW learning rate 1e-4/weight decay 1e-4/clip 1/seed 244691.

Reconstruct and hash-check all 1,121 audited augmentations before optimizer.
Choose the largest original Ns*Nt pair and the pair with the most synthetic null
labels, using original fitting order to break ties. Run exactly four updates:
original largest, augmented largest, original most-null, augmented most-null.
Keep original plus augmented supervision-count-derived square-root null weight,
using 20,293 parent and 1,458 null labels. No diagnostic-count-derived weights.

Replay complete original physical/neural diagnostics before optimizer but do not
evaluate model quality after these four smoke updates. Require finite loss and
gradient norm, nonzero head gradients, unchanged frozen tensors, and exact real
packet checkpoint reload. Record actual sample/augmentation identity and memory.
Only successful smoke permits separately frozen full 800-step 50/50 training.

Private offline two-T4 Kaggle runtime; 900-second hard cap, internal 780-second
worker deadline and 840-second watchdog. Fresh quota must preserve >=8h after
the full declared 0.25h. Sequential Biohub jobs only; no cloud/RSNA changes.
