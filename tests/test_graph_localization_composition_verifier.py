from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify-graph-localization-composition-candidate.py"
MODULE = runpy.run_path(str(SCRIPT))


def _promotion(run_id: str, runtime_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": run_id,
        "target_public_score": 0.945,
        "runtime_manifest_sha256": runtime_sha256,
        "submission_sha256": "a" * 64,
        "candidate_validator": {"proxy_score": 0.95},
        "known_public_hash_match": False,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
    }


def test_standalone_promotion_is_required_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "promotion.json"
    payload = _promotion("component-v1", "b" * 64)
    path.write_text(json.dumps(payload), encoding="utf-8")
    observed = MODULE["validate_component_promotion"](
        path,
        component="fixture",
        expected_run_id="component-v1",
        expected_runtime_sha256="b" * 64,
    )
    assert observed == payload
    payload["authorized_for_submission"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="standalone promotion"):
        MODULE["validate_component_promotion"](
            path,
            component="fixture",
            expected_run_id="component-v1",
            expected_runtime_sha256="b" * 64,
        )


def test_composition_gate_must_beat_both_components_without_metric_hacks() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert MODULE["MINIMUM_PROXY_GAIN"] == 0.005
    assert MODULE["MAXIMUM_ADJUSTED_EDGE_REGRESSION"] == 0.001
    assert MODULE["MINIMUM_COMPOSITION_GAIN_OVER_BEST_COMPONENT"] == 0.001
    assert "best_component_proxy = max(" in source
    assert "composition_gain >= MINIMUM_COMPOSITION_GAIN_OVER_BEST_COMPONENT" in source
    assert 'candidate_validator["division_jaccard"] >= graph_validator["division_jaccard"]' in source
    assert "node_nonregressive_to_localization" in source
    assert "public" in source
    assert "leaderboard" not in source.lower()
    assert "kaggle competitions submit" not in source

