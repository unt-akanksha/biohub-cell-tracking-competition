# Small Antelume frozen-head replay

Four existing cached training replay pairs only, from the same two three-frame
smoke clips. Transfer exact original19.2MB checkpoint, pinned model/runtime code,
packets and Kaggle neural-matrix references. Total stored archive20.28MB; no
new images/labels, detector weights, optimizer, source or target evaluation.

Run in a new bounded Biohub directory, never the RSNA project. Use existing
Python environment without installing/upgrading packages. Require GPU process
list empty before loading the model; do not stop other workloads. One CPU
thread, GPU allocation fraction capped0.2, outer timeout180s plus10s forced
exit grace. No overlapping Kaggle GPU experiment; feature/1 already COMPLETE.

Strict-load exact model tensors, FP32 highest matmul precision/TF32 disabled,
frozen eval mode. Compare four real neural matrices against original Kaggle
outputs, require deterministic within-host replay and unchanged final weights.
Record exact equality, maximum absolute error and real-parent argmax changes.
Do not automatically loosen original exact-replay requirements if hardware or
PyTorch version differs. Probe completion alone is not expanded-summary
acceptance, training evidence, source promotion or submission readiness.
