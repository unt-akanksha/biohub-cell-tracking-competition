# HOCT full-backprop spike

Date: 2026-08-27

Status: technically viable cloud fallback with trainer staged; no run launched.

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

An additional optimizer/save smoke used the parental objective intended for the
full run. One `AdamW` step reduced loss from `0.6708245` to `0.6687862`; the
adapted 25,503,532-byte ScriptModule reloaded with all 6,252,593 parameters and
reproduced loss `0.6687862` exactly.

## Decision

Keep this as the second cloud lane behind temporal appearance. A clean run must
use reciprocal embryo folds, corrected native synthetic geometry, the original
checkpoint as a selectable candidate, a very small full-backbone learning rate,
and L2-to-initialization regularization. It must not repeat the old linear-probe
conclusion or tune against the four processed acceptance movies.

The staged trainer implements those boundaries with exactly two reciprocal GPU
workers, 1,900 corrected synthetic graphs, conservative 10% real replay,
parental-softmax plus balanced edge loss, L2-SP anchoring, exact per-frame
candidate-recall checks, and the pretrained initialization retained as a
selectable checkpoint. The broader relevant test suite passes 54 tests.
