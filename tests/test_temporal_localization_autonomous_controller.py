from __future__ import annotations

from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/wait-stage-launch-verify-submit-temporal-localization-candidate.ps1"
SUBMITTER = runpy.run_path(str(ROOT / "scripts/submit-temporal-localization-candidate.py"))


def test_controller_is_event_driven_hash_bound_and_fail_closed() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "harvest-terminal.json" in source
    assert "Assert-SafeArchive" in source
    assert "result_sha256" in source
    assert "skipped_after_scientific_rejection" in source
    assert "MinimumGpuReserveHours = 0.0" in source
    assert "post_launch_state_retry" in source
    assert "DeclaredCandidateBudgetSeconds = 39600" in source
    assert "biohub-ct-0940-ema/validator_results.csv" in source
    assert "4dbf2079c1efc1108370f33104a6e80882a851d45dbfaa0540d40e88e4941f4b" in source
    assert "verify-temporal-localization-submission-candidate.py" in source
    assert "submit-temporal-localization-candidate.py" in source
    assert 'promotion.authorized_for_submission -ne $true' in source
    assert source.count("competitions submit") == 1
    assert "public_leaderboard_used_for_selection = $false" in source


def test_temporal_submitter_accepts_only_external_promotion(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,value\n1,x\n", encoding="utf-8")
    digest = SUBMITTER["sha256_file"](submission)
    promotion = {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": "ema-temporal-localization-candidate-v1",
        "target_public_score": 0.945,
        "known_public_hash_match": False,
        "localization_member_count": 4,
        "localization_nodes_moved": 10,
        "localization_rounded_coordinate_changes": 9,
        "localization_node_count_changes": 0,
        "localization_edge_changes": 0,
        "proxy_gain": 0.006,
        "adjusted_edge_delta": -0.0005,
        "missed_gt_node_gain": 2,
        "spurious_pred_node_delta": 0,
        "runtime_manifest_sha256": "a" * 64,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
        "submission_path": str(submission),
        "submission_sha256": digest,
    }
    path = tmp_path / "promotion.json"
    import json

    path.write_text(json.dumps(promotion), encoding="utf-8")
    observed, observed_submission = SUBMITTER["validate_promotion"](path)
    assert observed["proxy_gain"] == 0.006
    assert observed_submission == submission.resolve()

    promotion["missed_gt_node_gain"] = 0
    path.write_text(json.dumps(promotion), encoding="utf-8")
    with pytest.raises(RuntimeError, match="promotion evidence"):
        SUBMITTER["validate_promotion"](path)
