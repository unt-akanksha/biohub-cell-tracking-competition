from __future__ import annotations

import numpy as np

from research.synthetic_pretrain.data import POOLED_VOXEL_UM, SequenceSample
from research.temporal_localization.train_synthetic_localizer import (
    AUDIT_INDICES,
    SELECTION_INDICES,
    TRAIN_INDICES,
    baseline_metrics,
    build_sequence_state,
    fixed_examples,
    graph_motion_features,
    improvement_gate,
    random_jitter_um,
)


def fixture_state(index: int = 0):
    volumes = np.zeros((6, 8, 8, 8), dtype=np.uint16)
    nodes = np.asarray(
        [
            [1, 3, 3, 3, 1],
            [2, 3, 4, 3, 1],
            [2, 5, 5, 5, 2],
            [3, 3, 5, 3, 1],
        ],
        dtype=np.float32,
    )
    sample = SequenceSample(
        volumes=volumes,
        nodes=nodes,
        edges=np.asarray([[0, 1], [1, 3]], dtype=np.int64),
        divisions=np.empty((0,), dtype=np.int64),
        voxel_um=POOLED_VOXEL_UM.copy(),
    )
    return build_sequence_state(index, sample)


def test_splits_are_disjoint_and_audit_is_last() -> None:
    assert set(TRAIN_INDICES).isdisjoint(SELECTION_INDICES)
    assert set(TRAIN_INDICES).isdisjoint(AUDIT_INDICES)
    assert set(SELECTION_INDICES).isdisjoint(AUDIT_INDICES)
    assert max(SELECTION_INDICES) < min(AUDIT_INDICES)


def test_jitter_is_bounded_and_fixed_inventory_is_repeatable() -> None:
    jitter = random_jitter_um(np.random.default_rng(7), 1_000)
    assert np.linalg.norm(jitter, axis=1).max() <= 10.0 + 1e-5
    states = [fixture_state(12), fixture_state(13)]
    first = fixed_examples(states, seed=9, count=10)
    second = fixed_examples(states, seed=9, count=10)
    assert first.inventory_sha256 == second.inventory_sha256
    assert first.count == 10


def test_graph_features_encode_available_parent_child_context() -> None:
    state = fixture_state()
    rows = np.asarray([1, 2], dtype=np.int64)
    proposals = state.sample.nodes[rows, 1:4].copy()
    features = graph_motion_features(state, rows, proposals)
    assert features.shape == (2, 12)
    assert features[0, 6] == 1.0
    assert features[0, 7] == 1.0
    assert features[1, 6] == 0.0
    assert features[1, 7] == 0.0
    assert np.isfinite(features).all()


def test_gate_requires_large_mean_gain_and_axis_safety() -> None:
    examples = fixed_examples([fixture_state(12), fixture_state(13)], seed=11, count=20)
    baseline = baseline_metrics(examples)
    strong = {
        **baseline,
        "mean_residual_um": baseline["mean_residual_um"] * 0.5,
        "p90_residual_um": baseline["p90_residual_um"] * 0.6,
        "axis_mae_um": (np.asarray(baseline["axis_mae_um"]) * 0.5).tolist(),
        "within_5um_rate": min(1.0, baseline["within_5um_rate"] + 0.2),
    }
    assert improvement_gate(baseline, strong)["passed"] is True
    weak = {**strong, "mean_residual_um": baseline["mean_residual_um"] * 0.9}
    assert improvement_gate(baseline, weak)["passed"] is False
