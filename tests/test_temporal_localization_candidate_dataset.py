from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "scripts/build-temporal-localization-candidate-dataset.py"))
stage_dataset = MODULE["stage_dataset"]
verify_dataset = MODULE["verify_dataset"]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def result_fixture(root: Path, members: int = 3) -> Path:
    accepted = []
    for index in range(members):
        member = root / f"gpu_{index}" / f"member_00_seed_{41021 + index}"
        member.mkdir(parents=True)
        checkpoint = member / "localization_model.pt"
        checkpoint.write_bytes(f"independent-model-{index}".encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        terminal = {
            "schema_version": 1,
            "status": "completed",
            "run_id": "synthetic256-real-replay-temporal-node-localizer-v2",
            "parameter_count": 71_249_805,
            "seed": 41021 + index,
            "selection_gate_passed": True,
            "audit_gate_passed": True,
            "division_critical_selection_gate_passed": True,
            "division_critical_audit_gate_passed": True,
            "real_selection_gate_passed": True,
            "real_audit_gate_passed": True,
            "real_division_critical_selection_gate_passed": True,
            "real_division_critical_audit_gate_passed": True,
            "serialized_checkpoint_selection_gate_passed": True,
            "checkpoint_frozen_before_audit": True,
            "audit_opened": True,
            "model_sha256": digest,
            "best_selection": {"mean_residual_um": 1.1 + index / 10},
            "best_selection_division_critical": {
                "mean_residual_um": 1.0 + index / 10
            },
            "final_audit": {"mean_residual_um": 1.3 + index / 10},
            "final_audit_division_critical": {
                "mean_residual_um": 1.2 + index / 10
            },
            "best_real_selection": {"mean_residual_um": 1.05 + index / 10},
            "best_real_selection_division_critical": {
                "mean_residual_um": 1.0 + index / 10
            },
            "final_real_audit": {"mean_residual_um": 1.25 + index / 10},
            "final_real_audit_division_critical": {
                "mean_residual_um": 1.15 + index / 10
            },
            "real_replay_probability": 0.25,
            "competition_train_data_read": True,
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        write_json(member / "worker_terminal.json", terminal)
        accepted.append({"seed": 41021 + index, "model_sha256": digest})
    write_json(
        root / "real-development-probe.json",
        {
            "schema_version": 1,
            "status": "development_passed",
            "run_id": "temporal-node-localizer-real-development-v1",
            "training_run_id": "synthetic256-real-replay-temporal-node-localizer-v2",
            "members": accepted,
            "gate": {
                "passed": True,
                "no_movie_recall_regression": True,
                "matched_node_gain": 2,
                "mean_movie_matched_distance_improved": True,
                "all_global_move_fraction_gates_passed": True,
            },
            "acceptance_labels_already_opened": True,
            "development_only": True,
            "policy_or_member_selection_performed": False,
            "competition_test_data_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_submission": False,
        },
    )
    return root


def test_stage_and_verify_three_independently_strong_members(tmp_path: Path) -> None:
    output = tmp_path / "runtime"
    result = stage_dataset(result_fixture(tmp_path / "results"), output)
    assert result["status"] == "verified"
    assert result["localization_member_count"] == 3
    policy = json.loads((output / "temporal-localization-consensus-policy.json").read_text())
    assert policy["ensemble_policy"] == "equal_mean_all_dual_domain_eligible_members"
    assert policy["real_replay_probability"] == 0.25
    assert policy["maximum_move_fraction"] == 0.1
    assert policy["minimum_forced_division_critical_fraction"] == 0.25
    assert policy["division_critical_selection_gate_required"] is True
    assert policy["real_division_critical_audit_gate_required"] is True
    assert policy["authorized_for_submission"] is False


def test_runtime_verifier_detects_weight_tampering(tmp_path: Path) -> None:
    output = tmp_path / "runtime"
    stage_dataset(result_fixture(tmp_path / "results"), output)
    (output / "models/temporal_localizer_00.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="runtime changed"):
        verify_dataset(output)


def test_stage_rejects_fewer_than_three_members(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires 3 or 4"):
        stage_dataset(result_fixture(tmp_path / "results", members=2), tmp_path / "runtime")


def test_stage_rejects_member_without_serialized_stratum_gate(tmp_path: Path) -> None:
    results = result_fixture(tmp_path / "results")
    terminal_path = next(results.glob("gpu_*/member_*/worker_terminal.json"))
    terminal = json.loads(terminal_path.read_text())
    terminal["serialized_checkpoint_selection_gate_passed"] = False
    write_json(terminal_path, terminal)
    with pytest.raises(ValueError, match="accepted localization member evidence"):
        stage_dataset(results, tmp_path / "runtime")
