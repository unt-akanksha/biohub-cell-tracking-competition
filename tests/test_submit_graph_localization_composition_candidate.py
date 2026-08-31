from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(
    str(ROOT / "scripts/submit-graph-localization-composition-candidate.py")
)


def _promotion(submission: Path) -> dict:
    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": "ema-graph-localization-composition-v1",
        "target_public_score": 0.945,
        "submission_path": str(submission),
        "submission_sha256": MODULE["sha256_file"](submission),
        "graph_runtime_manifest_sha256": "a" * 64,
        "localization_runtime_manifest_sha256": "b" * 64,
        "graph_promotion_report_sha256": "c" * 64,
        "localization_promotion_report_sha256": "d" * 64,
        "graph_member_count": 4,
        "localization_member_count": 4,
        "ranked_edges_added": 3,
        "localization_nodes_moved": 20,
        "localization_rounded_coordinate_changes": 18,
        "proxy_gain": 0.009,
        "composition_gain_over_best_component": 0.0015,
        "adjusted_edge_delta": 0.002,
        "missed_gt_node_gain": 4,
        "spurious_pred_node_delta": 0,
        "known_public_hash_match": False,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
    }


def test_submitter_accepts_only_stronger_hash_bound_composition(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,value\n1,x\n", encoding="utf-8")
    promotion = _promotion(submission)
    path = tmp_path / "promotion.json"
    path.write_text(json.dumps(promotion), encoding="utf-8")
    observed, observed_submission = MODULE["validate_promotion"](path)
    assert observed["composition_gain_over_best_component"] == 0.0015
    assert observed_submission == submission.resolve()

    promotion["composition_gain_over_best_component"] = 0.0
    path.write_text(json.dumps(promotion), encoding="utf-8")
    with pytest.raises(RuntimeError, match="composition promotion evidence"):
        MODULE["validate_promotion"](path)


def test_submitter_contains_exactly_one_submission_command() -> None:
    source = (
        ROOT / "scripts/submit-graph-localization-composition-candidate.py"
    ).read_text(encoding="utf-8")
    assert source.count('"submit",') == 1
    assert "DAILY_SUBMISSION_LIMIT = 5" in source

