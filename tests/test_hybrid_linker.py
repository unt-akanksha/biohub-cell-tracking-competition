from __future__ import annotations

import numpy as np

from research.trackastra_graph.hybrid_linker import HybridLinkConfig, hybrid_link_pair


def test_locks_strong_base_edge_and_replaces_weak_edge() -> None:
    source_ids = np.array([10, 11])
    target_ids = np.array([20, 21])
    base = [(10, 20, 0.99), (11, 21, 0.55)]
    trackastra = np.array([[0.20, 0.95], [0.90, 0.10]], dtype=np.float32)

    result = hybrid_link_pair(
        source_ids,
        target_ids,
        trackastra,
        base,
        HybridLinkConfig(base_bonus=0.0),
    )

    assert set(result.edges) == {(10, 20), (11, 21)}
    assert result.locked_base_edges == 1
    # Target 20 remains occupied by the locked edge, so global constraints keep
    # the only valid remaining assignment rather than creating two parents.
    assert result.retained_base_edges == 2


def test_replaces_weak_base_assignment_when_targets_are_available() -> None:
    source_ids = np.array([10, 11])
    target_ids = np.array([20, 21])
    base = [(10, 20, 0.60), (11, 21, 0.60)]
    trackastra = np.array([[0.10, 0.90], [0.85, 0.10]], dtype=np.float32)

    result = hybrid_link_pair(
        source_ids,
        target_ids,
        trackastra,
        base,
        HybridLinkConfig(base_bonus=0.05),
    )

    assert set(result.edges) == {(10, 21), (11, 20)}
    assert result.retained_base_edges == 0
    assert result.new_primary_edges == 2


def test_adds_only_one_new_division_child_and_preserves_topology() -> None:
    source_ids = np.array([10, 11])
    target_ids = np.array([20, 21, 22])
    base = [(10, 20, 0.99), (11, 22, 0.99)]
    trackastra = np.array(
        [[0.90, 0.70, 0.02], [0.01, 0.05, 0.90]], dtype=np.float32
    )

    result = hybrid_link_pair(
        source_ids,
        target_ids,
        trackastra,
        base,
        HybridLinkConfig(division_threshold=0.50, division_ratio=0.50),
    )

    assert set(result.edges) == {(10, 20), (10, 21), (11, 22)}
    assert result.new_division_edges == 1
    assert max(sum(target == node for _, target in result.edges) for node in target_ids) == 1
    assert max(sum(source == node for source, _ in result.edges) for node in source_ids) == 2


def test_rejects_mismatched_score_shape() -> None:
    with np.testing.assert_raises(ValueError):
        hybrid_link_pair(
            np.array([1]),
            np.array([2, 3]),
            np.zeros((1, 1), dtype=np.float32),
            [],
            HybridLinkConfig(),
        )
