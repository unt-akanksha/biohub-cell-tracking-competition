from __future__ import annotations

from research.division_recovery_feasibility import (
    ranked_second_daughter_candidates,
)


def test_second_daughter_candidates_prioritize_parent_free_nodes() -> None:
    nodes = {
        1: (0, 0.0, 0.0, 0.0),
        2: (1, 0.0, 1.0, 0.0),
        3: (1, 0.0, 2.0, 0.0),
        4: (1, 0.0, 0.5, 0.0),
        5: (0, 0.0, 0.5, 0.0),
    }
    edges = [(1, 2), (5, 4)]

    candidates = ranked_second_daughter_candidates(
        nodes, edges, parent_id=1, existing_child_id=2
    )

    assert [row["node_id"] for row in candidates] == [3, 4]
    assert candidates[0]["parent_free"] is True
    assert candidates[1]["parent_free"] is False
