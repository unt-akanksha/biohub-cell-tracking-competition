# Checkpoint smoke: incomplete, scaling withheld

The recovered v2 checkpoint was strictly loaded into the actual organizer
TemporalUNet3D + association model on CPU. Inference ran on a three-frame
artificial 16x64x64 two-spot movie. This is not competition validation.

The resulting graph was empty. GEFF write/read preserved the node/edge counts,
but the current organizer scorer failed in division matching with `KeyError:
'z'` after copying the empty graph. Thus this is not a passed end-to-end
inference/serialization/scoring check. No metric source was changed and no
dummy graph nodes were inserted to conceal the failure.

Separately, a CPU regression reproduced nonfinite attention gradients when
one batch item has no real key nodes. The owned guard bypasses attention only
for those empty items; nonempty outputs and state-dict keys are unchanged.
The attention, split, builder and verifier tests total 15 passes. This guard
has not yet been tested in a new GPU pilot.

Next checks: locate exactly where empty spatial schema is lost, retain it
without changing nodes/edges or scoring rules, rerun the CPU smoke, then test
the guarded backward path in a small GPU pilot before any larger allocation.

## Follow-up: CPU smoke repaired and passed

The empty GEFF round trip dropped spatial attribute definitions. Restoring
the known `z/y/x` Float64 schema on an empty graph also preserves it through
the scorer's copy operation. The adapter refuses to invent coordinates for
a nonempty graph. Two real-library regression tests pass.

Actual-checkpoint inference, GEFF round trip, and unchanged organizer scoring
now complete: zero predicted nodes/edges, zero edge TP/FP, four edge FN on
the artificial movie. This verifies functionality, not detection quality.
Full result: `independent-real-checkpoint-smoke-result.json`. Guarded GPU
backward still requires a small pilot before scaling.
