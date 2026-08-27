# HOCT full-backprop spike

Date: 2026-08-27

Status: technically viable fallback; no training run launched.

## Question

The rejected HOCT experiment trained only a 289-parameter linear probe. The
official repository distributes JIT inference modules rather than a training
entry point, so it was unclear whether a genuine full-backbone Biohub
adaptation could be implemented from those artifacts.

## Result

The hash-pinned official `general_v1.pt` ScriptModule was switched to training
mode and run directly on a two-frame Biohub point window containing 8 nodes, 16
candidate edges, and 4 true edges. Binary link loss backpropagated successfully:

- total model parameters: 6,252,593;
- parameter tensors: 120;
- tensors with finite nonzero gradients: 114;
- parameters covered by those tensors: 6,245,681 (99.889%);
- initial smoke loss: 1.1052886;
- `input_proj.weight` gradient norm: 1.15318;
- first attention `q_proj.weight` gradient norm: 0.16340;
- first attention `kv_proj.weight` gradient norm: 0.27664.

This is direct evidence that full HOCT adaptation is possible without
reconstructing or copying an upstream architecture. The JIT model preserves
autograd through its input projection, positional encoding, attention blocks,
and classifier.

## Decision

Keep this as the second cloud lane behind temporal appearance. A clean run must
use reciprocal embryo folds, corrected native synthetic geometry, the original
checkpoint as a selectable candidate, a very small full-backbone learning rate,
and L2-to-initialization regularization. It must not repeat the old linear-probe
conclusion or tune against the four processed acceptance movies.
