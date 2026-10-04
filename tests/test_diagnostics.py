from __future__ import annotations

import json

import pytest

from biohub_tracker.diagnostics import (
    DiagnosticError,
    aggregate_movie_diagnostics,
    diagnose_movie,
)
from biohub_tracker.graphs import GraphData, GraphNode


def _graph(nodes, edges):
    return GraphData(
        nodes=tuple(GraphNode(*node) for node in nodes),
        edges=tuple(edges),
    )


TRUTH = _graph(
    (
        (0, 0, 0.0, 0.0, 0.0),
        (1, 1, 0.0, 0.0, 1.0),
        (2, 2, 0.0, 0.0, 3.0),
        (3, 2, 0.0, 0.0, 5.0),
    ),
    ((0, 1), (1, 2), (1, 3)),
)
PREDICTION = _graph(
    (
        (10, 0, 0.0, 0.0, 0.0),
        (11, 1, 0.0, 0.0, 1.0),
        (12, 2, 0.0, 0.0, 3.0),
        (13, 2, 0.0, 0.0, 5.0),
    ),
    ((10, 11), (11, 12)),
)
MATCHES = {10: 0, 11: 1, 12: 2, 13: 3}


def _diagnose(**overrides):
    values = {
        "prediction": PREDICTION,
        "truth": TRUTH,
        "prediction_to_truth": MATCHES,
        "official_counts": {
            "edge_tp": 2,
            "edge_fp": 0,
            "edge_fn": 1,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 1,
            "num_pred_nodes": 4,
        },
        "official_adjusted_edge_jaccard": "0.6666666666666666",
        "estimated_number_of_nodes": "4",
        "shape_tzyx": (3, 10, 10, 10),
        "scale_zyx_um": ("1", "1", "1"),
        "adjustment_alpha": "0.1",
        "displacement_boundaries_um": ("2", "4", "8"),
        "density_boundaries_per_mm3": ("1000", "2000", "3000"),
        "division_scores": {1: 0},
        "division_tp_forks": (),
        "division_fp_forks": (),
    }
    values.update(overrides)
    return diagnose_movie(**values)


def test_endpoint_oracle_and_conditional_counts_are_exact_and_reconciled():
    result = _diagnose()
    assert result["authority"] == "non_authoritative_diagnostic"
    assert result["organizer_input_eligible"] is False
    assert result["endpoint_availability"] == {
        "status": "applicable",
        "reason": None,
        "numerator": 3,
        "denominator": 3,
        "value": "1",
    }
    assert result["conditional_association_recall"]["value"] == "0.6666666666666666666666666667"
    assert result["conditional_valid_edge_precision"]["value"] == "1"
    assert result["conditional_valid_edge_jaccard"]["value"] == "0.6666666666666666666666666667"
    assert result["oracle_link_ceiling"]["raw"]["value"] == "1"
    assert result["oracle_link_ceiling"]["adjusted"]["value"] == "1"
    assert result["reconciliation"] == {
        "status": "passed",
        "edge_tp": 2,
        "edge_fp": 0,
        "edge_fn": 1,
        "division_tp": 0,
        "division_fp": 0,
        "division_fn": 1,
    }


def test_displacement_density_and_division_rows_cover_each_event_once():
    result = _diagnose()
    assert result["displacement"]["totals"] == {
        "gt_edges": 3,
        "endpoint_available": 3,
        "recovered": 2,
    }
    assert [row["bin"] for row in result["displacement"]["rows"]] == [
        "0_to_lt_2_um",
        "2_to_lt_4_um",
        "4_to_lt_8_um",
    ]
    assert sum(row["movie_count"] for row in result["density"]["rows"]) == 1
    categories = {row["category"]: row["count"] for row in result["divisions"]["rows"]}
    assert categories["missed_gt_division"] == 1
    assert sum(categories[f"support_offset_{offset}"] for offset in ("-1", "0", "+1")) == 0

    perfect_prediction = GraphData(
        PREDICTION.nodes, ((10, 11), (11, 12), (11, 13))
    )
    recovered = _diagnose(
        prediction=perfect_prediction,
        official_counts={
            "edge_tp": 3,
            "edge_fp": 0,
            "edge_fn": 0,
            "division_tp": 1,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 4,
        },
        official_adjusted_edge_jaccard="1",
        division_scores={1: 1},
        division_tp_forks=(11,),
    )
    recovered_categories = {
        row["category"]: row["count"] for row in recovered["divisions"]["rows"]
    }
    assert recovered_categories["support_offset_0"] == 1
    assert recovered["divisions"]["excluded_count"] == 0


def test_division_timing_uses_lineage_correspondence_not_temporal_greedy_pairing():
    truth = _graph(
        (
            (10, 1, 0.0, 0.0, 0.0),
            (11, 2, 0.0, 0.0, 1.0),
            (12, 2, 0.0, 0.0, 2.0),
            (20, 4, 0.0, 0.0, 3.0),
            (21, 5, 0.0, 0.0, 4.0),
            (22, 5, 0.0, 0.0, 5.0),
        ),
        ((10, 11), (10, 12), (20, 21), (20, 22)),
    )
    prediction = _graph(
        (
            (100, 1, 0.0, 0.0, 3.0),
            (101, 2, 0.0, 0.0, 4.0),
            (102, 2, 0.0, 0.0, 5.0),
            (200, 4, 0.0, 0.0, 0.0),
            (201, 5, 0.0, 0.0, 1.0),
            (202, 5, 0.0, 0.0, 2.0),
        ),
        ((100, 101), (100, 102), (200, 201), (200, 202)),
    )
    result = diagnose_movie(
        prediction=prediction,
        truth=truth,
        prediction_to_truth={100: 20, 101: 21, 102: 22, 200: 10, 201: 11, 202: 12},
        official_counts={
            "edge_tp": 4,
            "edge_fp": 0,
            "edge_fn": 0,
            "division_tp": 2,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 6,
        },
        official_adjusted_edge_jaccard="1",
        estimated_number_of_nodes="6",
        shape_tzyx=(6, 10, 10, 10),
        scale_zyx_um=("1", "1", "1"),
        adjustment_alpha="0.1",
        displacement_boundaries_um=("2", "4", "8"),
        density_boundaries_per_mm3=("1000", "2000", "3000"),
        division_scores={10: 1, 20: 1},
        division_tp_forks=(100, 200),
    )
    categories = {row["category"]: row["count"] for row in result["divisions"]["rows"]}
    assert categories["support_offset_0"] == 0
    assert categories["support_offset_outside_window"] == 2


def test_no_event_state_is_explicit_finite_and_aggregates_without_nan():
    graph = _graph(((0, 0, 0.0, 0.0, 0.0),), ())
    result = _diagnose(
        prediction=graph,
        truth=graph,
        prediction_to_truth={0: 0},
        official_counts={
            "edge_tp": 0,
            "edge_fp": 0,
            "edge_fn": 0,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 1,
        },
        official_adjusted_edge_jaccard="0",
        estimated_number_of_nodes="1",
        division_scores={},
    )
    assert result["endpoint_availability"]["status"] == "not_applicable"
    assert result["conditional_association_recall"]["status"] == "not_applicable"
    assert result["divisions"]["no_event"] is True
    aggregate = aggregate_movie_diagnostics((result, result))
    assert aggregate["endpoint_availability"]["status"] == "not_applicable"
    assert "NaN" not in json.dumps(aggregate, sort_keys=True)
    assert "Infinity" not in json.dumps(aggregate, sort_keys=True)


def test_diagnostic_reproduction_fails_closed_on_official_count_drift():
    with pytest.raises(DiagnosticError, match="DIAGNOSTIC_RECONCILIATION_FAILED"):
        _diagnose(
            official_counts={
                "edge_tp": 1,
                "edge_fp": 0,
                "edge_fn": 2,
                "division_tp": 0,
                "division_fp": 0,
                "division_fn": 1,
                "num_pred_nodes": 4,
            }
        )
