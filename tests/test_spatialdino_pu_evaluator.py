from __future__ import annotations

import hashlib
import json

import pytest

from research.spatialdino_detection.evaluate_pu_detector import (
    summarize_rows,
    validate_training_result,
)


def test_training_result_requires_independent_completed_ema(tmp_path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"learned")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "ema_checkpoint": True,
                "validation_overlap": [],
                "public_predictions_copied": False,
                "best_weight_sha256": hashlib.sha256(b"learned").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert validate_training_result(model, result)["status"] == "completed"
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["public_predictions_copied"] = True
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="independent"):
        validate_training_result(model, result)


def test_summary_pools_movies_and_embryo_prefixes() -> None:
    rows = [
        {
            "stem": "44b6_a",
            "candidate": {"annotated_gt_nodes": 10, "matched_gt_nodes": 9, "annotated_node_recall": 0.9},
        },
        {
            "stem": "6bba_b",
            "candidate": {"annotated_gt_nodes": 20, "matched_gt_nodes": 16, "annotated_node_recall": 0.8},
        },
    ]
    summary = summarize_rows(rows)
    assert summary["annotated_node_recall"] == 25 / 30
    assert summary["worst_movie_recall"] == 0.8
    assert summary["by_prefix"]["44b6"]["annotated_node_recall"] == 0.9
