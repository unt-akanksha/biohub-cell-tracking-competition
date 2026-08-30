from __future__ import annotations

import numpy as np
import pytest

from biohub_tracker.graphs import GraphData, GraphNode
from research.temporal_localization.consensus import (
    LocalizationConsensusPolicy,
    apply_physical_coordinate_offsets,
    select_consensus_offsets,
)


def test_consensus_selects_only_agreed_unsafe_low_uncertainty_offsets() -> None:
    offsets = np.asarray(
        [
            [[6.0, 0.0, 0.0], [6.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[6.2, 0.1, 0.0], [-6.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[5.8, -0.1, 0.0], [6.1, 0.0, 0.0], [0.9, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    sigma = np.ones_like(offsets)
    safe = np.full(offsets.shape[:2], 0.1, dtype=np.float32)
    result = select_consensus_offsets(
        offsets, sigma, safe, policy=LocalizationConsensusPolicy(maximum_move_fraction=0.5)
    )
    assert result.selected.tolist() == [True, False, False]
    assert result.offsets_um[0, 0] == pytest.approx(4.5)
    assert result.report["member_subset_search"] is False


def test_consensus_global_fraction_gate_fails_closed() -> None:
    offsets = np.tile(np.asarray([[[6.0, 0.0, 0.0]]], dtype=np.float32), (3, 20, 1))
    sigma = np.ones_like(offsets)
    safe = np.full((3, 20), 0.1, dtype=np.float32)
    result = select_consensus_offsets(offsets, sigma, safe)
    assert result.global_gate_passed is False
    assert not result.selected.any()
    assert not result.offsets_um.any()


def test_consensus_requires_three_independently_accepted_members() -> None:
    with pytest.raises(ValueError, match="too few"):
        select_consensus_offsets(
            np.zeros((2, 1, 3), dtype=np.float32),
            np.ones((2, 1, 3), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
        )


def test_coordinate_application_preserves_node_and_edge_inventory() -> None:
    control = GraphData(
        nodes=(
            GraphNode(10, 0, 2.0, 3.0, 4.0),
            GraphNode(11, 1, 3.0, 4.0, 5.0),
        ),
        edges=((10, 11),),
    )
    refined = apply_physical_coordinate_offsets(
        control,
        {11: np.asarray((1.625, 0.40625, -0.40625), dtype=np.float32)},
        spatial_shape_zyx=(10, 10, 10),
    )
    assert refined.edges == control.edges
    assert [(node.node_id, node.t) for node in refined.nodes] == [(10, 0), (11, 1)]
    assert (refined.nodes[1].z, refined.nodes[1].y, refined.nodes[1].x) == pytest.approx((4, 5, 4))
    assert refined.nodes[0] == control.nodes[0]


def test_coordinate_application_rejects_boundary_exit() -> None:
    control = GraphData(nodes=(GraphNode(1, 0, 0.0, 0.0, 0.0),), edges=())
    with pytest.raises(ValueError, match="leaves the image"):
        apply_physical_coordinate_offsets(
            control,
            {1: np.asarray((-1.0, 0.0, 0.0), dtype=np.float32)},
            spatial_shape_zyx=(10, 10, 10),
        )
