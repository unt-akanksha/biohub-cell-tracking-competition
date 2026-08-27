from __future__ import annotations

import pytest

from research.lsm_fm_detection.materialize_public_node_candidate import (
    assert_raw_topology_compatible,
)


def test_raw_topology_accepts_coordinate_only_candidate() -> None:
    assert_raw_topology_compatible(
        {1: 0, 2: 1},
        [(1, 2)],
        {1: 0, 2: 1},
        [(1, 2)],
        stem="movie",
    )


@pytest.mark.parametrize(
    ("candidate_nodes", "candidate_edges", "message"),
    [
        ({1: 0, 2: 2}, [(1, 2)], "node IDs or frame"),
        ({1: 0, 2: 1}, [(2, 1)], "changed public raw topology"),
    ],
)
def test_raw_topology_rejects_non_coordinate_changes(
    candidate_nodes, candidate_edges, message
) -> None:
    with pytest.raises(RuntimeError, match=message):
        assert_raw_topology_compatible(
            {1: 0, 2: 1},
            [(1, 2)],
            candidate_nodes,
            candidate_edges,
            stem="movie",
        )
