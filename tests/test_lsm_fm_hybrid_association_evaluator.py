from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.lsm_fm_detection.evaluate_hybrid_association import (
    canonical_graph_sha256,
    prefix_summaries,
    selected_detector,
)


def _selection_payload(**changes):
    payload = {
        "status": "completed",
        "selection_passed": True,
        "acceptance_opened": True,
        "promotion_passed": True,
        "selected_candidate": "equal_ensemble_log_quadratic",
        "public_leaderboard_used_for_selection": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
    }
    payload.update(changes)
    return payload


def test_selected_detector_requires_complete_clean_promotion(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_selection_payload()), encoding="utf-8")
    candidate, _ = selected_detector(path)
    assert candidate.name == "equal_ensemble_log_quadratic"
    assert candidate.refinement == "log_quadratic"

    path.write_text(
        json.dumps(_selection_payload(promotion_passed=False)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="clean node acceptance"):
        selected_detector(path)


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("public_leaderboard_used_for_selection", True, "not clean"),
        ("public_predictions_copied", True, "copied"),
        ("competition_submission_performed", True, "submitted"),
    ],
)
def test_selected_detector_rejects_contaminated_provenance(
    tmp_path: Path, field: str, value: bool, pattern: str
) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(_selection_payload(**{field: value})), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=pattern):
        selected_detector(path)


class _Rows:
    def __init__(self, values):
        self._values = values

    def iter_rows(self, named=True):
        assert named
        return iter(self._values)


class _Graph:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    def node_attrs(self):
        return _Rows(self._nodes)

    def edge_attrs(self):
        return _Rows(self._edges)


def test_canonical_graph_hash_ignores_storage_row_order() -> None:
    nodes = [
        {"node_id": 2, "t": 1, "z": 2.0, "y": 3.0, "x": 4.0},
        {"node_id": 1, "t": 0, "z": 1.0, "y": 2.0, "x": 3.0},
    ]
    edges = [{"source_id": 1, "target_id": 2}]
    first = canonical_graph_sha256(_Graph(nodes, edges))
    second = canonical_graph_sha256(_Graph(list(reversed(nodes)), edges))
    assert first == second
    changed = [dict(nodes[0]), dict(nodes[1])]
    changed[0]["x"] = 4.01
    assert canonical_graph_sha256(_Graph(changed, edges)) != first


class _Metrics:
    @staticmethod
    def summarise(rows):
        return {"names": [row["name"] for row in rows], "score": len(rows)}


def test_prefix_summaries_do_not_mix_embryos() -> None:
    rows = [
        {"name": "44b6_a"},
        {"name": "44b6_b"},
        {"name": "6bba_a"},
    ]
    result = prefix_summaries(rows, _Metrics())
    assert result["44b6"]["names"] == ["44b6_a", "44b6_b"]
    assert result["6bba"]["names"] == ["6bba_a"]
