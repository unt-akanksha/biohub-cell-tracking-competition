from __future__ import annotations

import pytest

from research.build_competition_division_probe_inventory import candidate_geometry_features


def test_candidate_geometry_captures_symmetric_opposed_daughters() -> None:
    nodes = {
        1: {"t": 4, "z": 1.0, "y": 1.0, "x": 1.0},
        2: {"t": 5, "z": 1.0, "y": 3.0, "x": 1.0},
        3: {"t": 5, "z": 1.0, "y": -1.0, "x": 1.0},
        4: {"t": 3, "z": 1.0, "y": 0.0, "x": 1.0},
    }

    features = candidate_geometry_features(
        nodes,
        {1: [4]},
        parent_id=1,
        existing_child_id=2,
        second_child_id=3,
    )

    assert features["daughter_midpoint_distance_um"] == pytest.approx(0.0)
    assert features["daughter_opposition_cosine"] == pytest.approx(-1.0)
    assert features["daughter_step_ratio"] == pytest.approx(1.0)
    assert features["predecessor_id"] == 4
    assert features["constant_velocity_midpoint_error_um"] == pytest.approx(0.40625)


def test_candidate_geometry_handles_missing_unique_predecessor() -> None:
    nodes = {
        1: {"t": 4, "z": 0.0, "y": 0.0, "x": 0.0},
        2: {"t": 5, "z": 0.0, "y": 1.0, "x": 0.0},
        3: {"t": 5, "z": 0.0, "y": 0.0, "x": 1.0},
    }

    features = candidate_geometry_features(
        nodes, {}, parent_id=1, existing_child_id=2, second_child_id=3
    )

    assert features["predecessor_id"] is None
    assert features["parent_velocity_um"] is None
    assert features["constant_velocity_midpoint_error_um"] is None
