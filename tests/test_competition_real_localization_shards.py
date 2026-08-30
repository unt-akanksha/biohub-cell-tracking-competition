from __future__ import annotations

import numpy as np
import pytest

from research.build_competition_real_localization_shards import (
    division_critical_center_rows,
    triplet_payload,
)


def fixture_graph():
    nodes = {
        10: (4.0, 10.0, 40.0, 44.0),
        20: (5.0, 11.0, 44.0, 48.0),
        30: (6.0, 12.0, 48.0, 52.0),
        31: (6.0, 9.0, 36.0, 40.0),
    }
    edges = [(10, 20), (20, 30), (20, 31)]
    return nodes, edges


def test_triplet_is_pooled_geometry_safe_and_lineage_preserving() -> None:
    nodes, edges = fixture_graph()
    frames = {
        frame: np.full((64, 256, 256), frame, dtype=np.uint16)
        for frame in (4, 5, 6)
    }
    payload = triplet_payload(nodes, edges, 5, frames.__getitem__)
    assert payload["volumes"].shape == (3, 64, 64, 64)
    assert payload["volumes"].dtype == np.uint16
    assert payload["nodes"].shape == (4, 5)
    assert payload["nodes"][:, 0].tolist() == [0.0, 1.0, 2.0, 2.0]
    assert payload["edges"].tolist() == [[0, 1], [1, 2], [1, 3]]
    assert payload["divisions"].tolist() == [1]
    assert division_critical_center_rows(payload) == 1


def test_triplet_rejects_wrong_image_geometry() -> None:
    nodes, edges = fixture_graph()
    with pytest.raises(ValueError, match="frame shape"):
        triplet_payload(
            nodes,
            edges,
            5,
            lambda _frame: np.zeros((64, 64, 64), dtype=np.uint16),
        )


def test_triplet_rejects_missing_center_labels() -> None:
    nodes, edges = fixture_graph()
    shifted = {node_id: (time + 10, z, y, x) for node_id, (time, z, y, x) in nodes.items()}
    with pytest.raises(ValueError, match="no center-frame labels"):
        triplet_payload(
            shifted,
            edges,
            5,
            lambda _frame: np.zeros((64, 256, 256), dtype=np.uint16),
        )
