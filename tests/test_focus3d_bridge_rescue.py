from __future__ import annotations

from research.focus3d_bridge_rescue import apply_focus_bridge
import pytest


def node(node_id: int, t: int, x: float) -> dict:
    return {"node_id": node_id, "t": t, "z": 10.0, "y": 10.0, "x": x}


def test_adds_one_confirmed_single_frame_bridge() -> None:
    base_nodes = {0: node(0, 0, 0.0), 1: node(1, 2, 8.0)}
    focus_nodes = {10: node(10, 0, 0.0), 11: node(11, 1, 4.0), 12: node(12, 2, 8.0)}
    focus_edges = [{"source_id": 10, "target_id": 11}, {"source_id": 11, "target_id": 12}]
    nodes, edges, stats = apply_focus_bridge(base_nodes, [], focus_nodes, focus_edges, {11: 0.8})
    assert len(nodes) == 3 and len(edges) == 2
    assert stats == {
        "eligible_bridges": 1, "added_nodes": 1, "added_edges": 2,
        "reassignment_performed": 0, "existing_node_or_edge_changes": 0,
    }
    assert nodes[2]["focus_source_node_id"] == 11


def test_negative_smoothed_support_does_not_abort_valid_bridge():
    base = {0: node(0, 0, 0), 1: node(1, 2, 8)}
    focus = {10: node(10, 0, -0.2), 11: node(11, 1, 4), 12: node(12, 2, 8)}
    edges = [{'source_id': 10, 'target_id': 11}, {'source_id': 11, 'target_id': 12}]
    result, _, stats = apply_focus_bridge(base, [], focus, edges, {11: .8})
    assert stats['added_nodes'] == 1 and result[2]['x'] == 4
    focus[11]['x'] = -0.1
    result, _, stats = apply_focus_bridge(base, [], focus, edges, {11: .8})
    assert stats['added_nodes'] == 0 and result == base


def test_rejects_unconfirmed_or_already_detected_focus_node() -> None:
    base_nodes = {0: node(0, 0, 0.0), 1: node(1, 1, 4.0), 2: node(2, 2, 8.0)}
    focus_nodes = {10: node(10, 0, 0.0), 11: node(11, 1, 4.0), 12: node(12, 2, 8.0)}
    focus_edges = [{"source_id": 10, "target_id": 11}, {"source_id": 11, "target_id": 12}]
    nodes, edges, stats = apply_focus_bridge(base_nodes, [], focus_nodes, focus_edges, {11: 0.9})
    assert nodes == base_nodes and edges == [] and stats["added_nodes"] == 0
    nodes, edges, stats = apply_focus_bridge(
        {0: node(0, 0, 0.0), 2: node(2, 2, 8.0)}, [], focus_nodes, focus_edges, {11: 0.24}
    )
    assert len(nodes) == 2 and edges == [] and stats["added_nodes"] == 0


def test_accepts_serialized_string_focus_ids() -> None:
    base_nodes = {0: node(0, 0, 0.0), 1: node(1, 2, 8.0)}
    focus_nodes = {"10": node(10, 0, 0.0), "11": node(11, 1, 4.0), "12": node(12, 2, 8.0)}
    focus_edges = [{"source_id": 10, "target_id": 11}, {"source_id": 11, "target_id": 12}]
    nodes, edges, stats = apply_focus_bridge(base_nodes, [], focus_nodes, focus_edges, {"11": 0.8})
    assert len(nodes) == 3 and len(edges) == 2 and stats["added_nodes"] == 1


@pytest.mark.parametrize("bad_edge", [
    {"source_id": 10, "target_id": 12},
    {"source_id": 99, "target_id": 11},
    {"source_id": 12, "target_id": 11},
])
def test_rejects_invalid_focus_support(bad_edge) -> None:
    base = {0: node(0, 0, 0), 1: node(1, 2, 8)}
    focus = {10: node(10, 0, 0), 11: node(11, 1, 4), 12: node(12, 2, 8)}
    with pytest.raises(ValueError):
        apply_focus_bridge(base, [], focus, [bad_edge], {11: 0.8})


def test_nonfinite_coordinate_rejected() -> None:
    with pytest.raises(ValueError):
        apply_focus_bridge({0: node(0, 0, float("nan"))}, [], {}, [], {})


def test_frozen_tie_break_uses_focus_id_before_endpoint_distance() -> None:
    base = {0: node(0, 0, 0), 1: node(1, 2, 8),
            2: node(2, 0, 100), 3: node(3, 2, 104)}
    focus = {10: node(10, 1, 4), 11: node(11, 0, 0), 12: node(12, 2, 8),
             20: node(20, 1, 102), 21: node(21, 0, 100), 22: node(22, 2, 104)}
    edges = [{"source_id": 11, "target_id": 10}, {"source_id": 10, "target_id": 12},
             {"source_id": 21, "target_id": 20}, {"source_id": 20, "target_id": 22}]
    nodes, _, _ = apply_focus_bridge(base, [], focus, edges, {10: 0.8, 20: 0.8})
    assert nodes[4]["focus_source_node_id"] == 10
