# Two training/inference hypotheses screened without a GPU launch

## Detector-produced association training already exists

The pinned organizer `train_unet_transformer.py` training loop calls
`detect_and_match` on predicted raw-logit peaks and builds matched edge targets.
`run-independent-association-pilot.py` preserves that call through its guarded
wrapper. The retained linker was not trained only on ground-truth coordinates.
Do not present predicted-node training as an unimplemented new method.

There are remaining distribution differences (native AMP training versus FP32
D4 inference), but their impact has not been isolated by this audit.

## Nearest-only matching is not a demonstrated bottleneck

Training uses a5-micron match radius and greedily assigns a detection only to
its closest annotated cell. In principle, a second valid annotated neighbor
could be ignored after the closest one is taken. A four-test CPU diagnostic
compares this rule with maximum-cardinality, minimum-distance one-to-one
matching at the same radius; a synthetic example demonstrates the difference.

On all360 previously verified training-frame artifacts (120 training movies),
the parent D4 detections have3304 annotations and84563 predicted peaks.
Both rules match2946 annotations. There are zero additional matches, zero
changed assignments and zero detections within5 microns of multiple annotations.
Result: `training-node-matching-diagnostic-v1.json`.

These are cached parent FP32 D4 detections, not bitwise replay of native AMP
training or its tie order. Nevertheless, there is no measured supervision gain
to justify a GPU training run for this matching change. No training, graph,
metric or threshold was changed; failed sparse calibration stays rejected.

## Temporal reversal alone is redundant for this detector architecture

Inspection of the pinned `models/temporal_unet.py` shows spatial convolutions
shared across frames and per-voxel temporal self-attention without temporal
positional encoding, causal masks or time-specific parameters. In evaluation
mode, reversing the same input-frame pair and reversing the outputs is
mathematically permutation-equivariant (apart from numerical effects).
It is not a distinct temporal-context ensemble. This is a source-derived
inference, not a measured checkpoint result. Do not spend GPU on merely
reversing the same pair; using different neighboring frames would be a
different, separately testable hypothesis.

Source inspected: official vendor commit075fc5f5a52d11077f9dc2b074644618f26939e2.
No new target movies, external models or GPU jobs were opened for this audit.
