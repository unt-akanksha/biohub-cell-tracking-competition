from __future__ import annotations

import math

from research.lsm_fm_detection.score_public_node_refinement import (
    finite_float,
    metric_delta,
)


def test_finite_float_maps_undefined_division_metric_to_none() -> None:
    assert finite_float(float("nan")) is None
    assert finite_float(0.25) == 0.25


def test_metric_delta_keeps_missing_division_explicit() -> None:
    control = {
        "score": 0.90,
        "adjusted_edge_jaccard": 0.88,
        "edge_jaccard": 0.87,
        "division_jaccard": None,
        "node_recall_micro": 0.95,
    }
    candidate = {
        "score": 0.91,
        "adjusted_edge_jaccard": 0.89,
        "edge_jaccard": 0.88,
        "division_jaccard": None,
        "node_recall_micro": 0.96,
    }
    result = metric_delta(candidate, control)
    assert math.isclose(result["score"], 0.01)
    assert math.isclose(result["node_recall_micro"], 0.01)
    assert result["division_jaccard"] is None
