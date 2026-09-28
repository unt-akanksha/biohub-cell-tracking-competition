"""A minimum learned-edge-probability floor for motion relinking.

WHY THIS AND NOT A DISTANCE GATE

The 40-movie error decomposition (nodecount-sweep-v1, base config) says the
dominant failure is false-positive edges, not missed ones:

    edge_tp 22374   edge_fp 1207   edge_fn 1211
    corr(adjusted edge Jaccard, fp rate)    = -0.929
    corr(adjusted edge Jaccard, frag rate)  = -0.806
    corr(adjusted edge Jaccard, det rate)   = -0.806

and the false-positive rate scales roughly five-fold with density, while the
detection-loss rate is nearly flat:

    sparsest 10 movies   density  51-98    fp/weight 0.0176   adjJ 0.9561
    10-20                density  98-221   fp/weight 0.0454   adjJ 0.9022
    20-30                density 225-331   fp/weight 0.0530   adjJ 0.9030
    densest 10           density 387-747   fp/weight 0.0865   adjJ 0.8517

So the problem is wrong links between nearby cells in dense frames. That is an
association failure, not a detection failure.

THE MECHANISM

`motion_relink_edges` rebuilds essentially the entire output graph -- run_stats
shows motion_relink_replaced_raw_edges at or above the final edge count on every
movie. Its assignment cost is

    cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob

and a pair is admitted purely on `raw > gate_um`. There is NO minimum on `prob`.
The Hungarian assignment therefore matches every source inside the gate to some
target, no matter how improbable the learned model considers the link. A cell
whose true successor is absent -- missed by the detector, left the field of
view, or genuinely the end of a track -- is still matched to a neighbour. Dense
frames have more candidates inside the gate, so they manufacture more of these,
which is exactly the five-fold scaling above.

Worse, `learned_edge_probs.get((source, target), 0.0)` returns 0.0 for any pair
the transformer never proposed, so relink can and does invent links the learned
model never suggested at all.

WHY THIS AXIS IS DIFFERENT FROM EVERYTHING ALREADY TRIED

Six hypotheses have failed on this pipeline: the D4 correction, track recovery,
the 165 band boundary, continuous tight gates, loosen-only gates, and node-count
calibration; the wide and loose sweeps were flat within 0.0009. Every one of
those moved a DISTANCE threshold or a track-length threshold. A distance gate
cannot separate a correct near pair from an incorrect near pair -- in a dense
frame both are near. The learned probability can. That is the whole argument for
expecting a different answer here, and it is a mechanism rather than a hope.

CALIBRATION OF THE FLOOR

The predictor already discards candidates below its own inference threshold
(BIOHUB_DUAL_SEED_EDGE_THRESHOLD = 0.48), so every pair that survived as a raw
edge carries prob > 0.48, while every pair the model did not propose carries
exactly 0.0. The floor therefore has two distinct regimes:

    floor = 0.0          current behaviour, no constraint
    0 < floor <= 0.48    admit only pairs the transformer actually proposed
    floor > 0.48         additionally require confidence among those

Both regimes are worth probing, and they are probing different claims.

This module defaults the floor to 0.0, which is a no-op, so a production build
is byte-identical in behaviour to the shipped 0.949 configuration unless the
floor is deliberately raised.
"""
from __future__ import annotations

import ast

# Shipped default. 0.0 reproduces current behaviour exactly.
MIN_LEARNED_PROB_DEFAULT = 0.0

SWEEPABLE_GLOBALS = ("MOTION_RELINK_MIN_LEARNED_PROB",)

_DEF_ANCHOR = "def motion_relink_edges(\n"
_PROB_ANCHOR = "                prob = learned_prob(source_id, target_id)\n"
_SWEEP_KEYS_ANCHOR = '    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM",\n]\n'


def preamble(min_prob: float = MIN_LEARNED_PROB_DEFAULT) -> str:
    return "\n".join([
        "",
        "",
        "# ---------------------------------------------------------------------------",
        "# Minimum learned edge probability for a motion-relink match.",
        "#",
        "# The relink assignment admits any pair inside the distance gate, so a cell",
        "# whose true successor is absent is still matched to a neighbour. Pairs the",
        "# transformer never proposed score exactly 0.0 here, so any floor above zero",
        "# restricts relinking to model-proposed pairs.",
        "#",
        "# 0.0 is a no-op and is what production ships.",
        "# ---------------------------------------------------------------------------",
        f"MOTION_RELINK_MIN_LEARNED_PROB = {float(min_prob)!r}",
        "",
    ])


def install(cell_source: str, min_prob: float = MIN_LEARNED_PROB_DEFAULT) -> tuple[str, dict]:
    """Define the floor and enforce it inside the assignment. Fails closed."""
    for name, anchor in (("definition", _DEF_ANCHOR), ("probability", _PROB_ANCHOR)):
        if cell_source.count(anchor) != 1:
            raise ValueError(f"Edge-confidence {name} anchor is not unique")

    patched = cell_source.replace(
        _DEF_ANCHOR, preamble(min_prob) + "\n" + _DEF_ANCHOR, 1
    )
    # Skip the pair entirely: leaving cost at `big` means linear_sum_assignment
    # cannot select it, and the existing `cost >= big` check drops it.
    guard = (
        "                if prob < MOTION_RELINK_MIN_LEARNED_PROB:\n"
        "                    continue\n"
    )
    patched = patched.replace(_PROB_ANCHOR, _PROB_ANCHOR + guard, 1)
    ast.parse(patched)

    report = {
        "min_learned_prob_default": float(min_prob),
        "is_no_op_by_default": float(min_prob) == 0.0,
        "sweepable_globals": list(SWEEPABLE_GLOBALS),
        "targets": "false-positive edges, the dominant error mode at corr -0.929",
        "inference_threshold_context": 0.48,
        "unproposed_pair_probability": 0.0,
        "added_lines": len(patched.splitlines()) - len(cell_source.splitlines()),
    }
    return patched, report


def register_sweep_keys(cell_source: str) -> str:
    """Let the notebook's own sweep rebind the floor. Fails closed.

    Valid only because the preamble lands in an earlier cell than the one that
    builds PP_BASE_CONFIG by reading each name out of globals().
    """
    if cell_source.count(_SWEEP_KEYS_ANCHOR) != 1:
        raise ValueError("PP_SWEEP_KEYS anchor is not unique")
    added = "    " + ", ".join(repr(n) for n in SWEEPABLE_GLOBALS) + ",\n"
    patched = cell_source.replace(
        _SWEEP_KEYS_ANCHOR, _SWEEP_KEYS_ANCHOR[:-2] + added + "]\n", 1
    )
    ast.parse(patched)
    for name in SWEEPABLE_GLOBALS:
        if patched.count(repr(name)) != 1:
            raise ValueError(f"{name} is not registered exactly once")
    return patched
