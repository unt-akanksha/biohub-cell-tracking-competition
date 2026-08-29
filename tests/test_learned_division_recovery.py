from __future__ import annotations

from research.learned_division_recovery import (
    DivisionRecoveryPolicy,
    apply_learned_division_recovery,
    discover_division_recovery_candidates,
)


def node(node_id: int, timepoint: int, y: float) -> tuple[int, dict]:
    return node_id, {"node_id": node_id, "t": timepoint, "z": 0.0, "y": y, "x": 0.0}


def test_discovery_selects_nearest_parent_free_second_daughter() -> None:
    nodes = dict(
        [
            node(1, 0, 0.0),
            node(2, 1, 2.0),
            node(3, 1, 3.0),
            node(4, 1, 1.0),
            node(5, 0, 100.0),
        ]
    )
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 5, "target_id": 4},
    ]

    candidates = discover_division_recovery_candidates(nodes, edges)

    assert len(candidates) == 1
    assert candidates[0].parent_id == 1
    assert candidates[0].second_child_id == 3


def test_learned_gate_adds_only_high_score_parent_and_preserves_nodes() -> None:
    nodes = dict(
        [
            node(1, 0, 0.0),
            node(2, 1, 2.0),
            node(3, 1, 3.0),
            node(10, 0, 20.0),
            node(11, 1, 22.0),
            node(12, 1, 23.0),
        ]
    )
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 10, "target_id": 11},
    ]
    policy = DivisionRecoveryPolicy(
        division_logit_threshold=1.0,
        maximum_added_node_fraction=0.01,
    )

    output, stats = apply_learned_division_recovery(
        nodes, edges, {1: 2.0, 10: 0.5}, policy
    )

    assert [(edge["source_id"], edge["target_id"]) for edge in output] == [
        (1, 2),
        (10, 11),
        (1, 3),
    ]
    assert output[-1]["learned_division_recovery"] is True
    assert stats["added_edges"] == 1
    assert stats["reassignment_performed"] == 0
    assert stats["node_or_coordinate_changes"] == 0


def test_recovery_never_steals_a_child_with_an_existing_parent() -> None:
    nodes = dict(
        [
            node(1, 0, 0.0),
            node(2, 1, 2.0),
            node(3, 1, 3.0),
            node(4, 0, 4.0),
        ]
    )
    edges = [
        {"source_id": 1, "target_id": 2},
        {"source_id": 4, "target_id": 3},
    ]

    output, stats = apply_learned_division_recovery(
        nodes,
        edges,
        {1: 5.0},
        DivisionRecoveryPolicy(division_logit_threshold=1.0),
    )

    assert output == edges
    assert stats["added_edges"] == 0
