from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.temporal_localization.plan_member_rescue import (
    RESCUE_SEEDS,
    plan_rescue,
)


def write_member(root: Path, gpu_index: int, seed: int, *, accepted: bool) -> None:
    member = root / f"gpu_{gpu_index}" / f"member_00_seed_{seed}"
    member.mkdir(parents=True)
    terminal = {
        "schema_version": 1,
        "run_id": "synthetic256-real-replay-temporal-node-localizer-v2",
        "status": "rejected_at_selection",
        "seed": seed,
    }
    if accepted:
        checkpoint = member / "localization_model.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        terminal.update(
            {
                "status": "completed",
                "parameter_count": 71_249_805,
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
                "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "real_replay_probability": 0.25,
                "competition_train_data_read": True,
                "competition_test_data_read": False,
                "public_code_copied": False,
                "public_predictions_copied": False,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
            }
        )
    (member / "worker_terminal.json").write_text(
        json.dumps(terminal), encoding="utf-8"
    )


def test_rescue_uses_fixed_seed_order_until_three_members_pass(tmp_path: Path) -> None:
    write_member(tmp_path, 0, 41_021, accepted=True)
    write_member(tmp_path, 1, 41_029, accepted=False)
    write_member(tmp_path, 2, 41_039, accepted=True)
    write_member(tmp_path, 3, 41_047, accepted=False)

    first = plan_rescue(tmp_path)
    assert first["status"] == "train_next_fixed_seed"
    assert first["next_seed"] == RESCUE_SEEDS[0]
    assert first["next_gpu_index"] == 4
    assert first["member_or_policy_ranking_performed"] is False

    write_member(tmp_path, 4, RESCUE_SEEDS[0], accepted=False)
    second = plan_rescue(tmp_path)
    assert second["next_seed"] == RESCUE_SEEDS[1]
    assert second["next_gpu_index"] == 5

    write_member(tmp_path, 5, RESCUE_SEEDS[1], accepted=True)
    terminal = plan_rescue(tmp_path)
    assert terminal["status"] == "sufficient_members"
    assert terminal["accepted_member_count"] == 3
    assert terminal["next_seed"] is None
    assert terminal["public_leaderboard_used_for_selection"] is False


def test_interrupted_rescue_seed_is_not_retried_or_ranked(tmp_path: Path) -> None:
    write_member(tmp_path, 0, 41_021, accepted=True)
    partial = tmp_path / "gpu_4" / f"member_00_seed_{RESCUE_SEEDS[0]}"
    partial.mkdir(parents=True)

    plan = plan_rescue(tmp_path)
    assert plan["attempted_rescue_seeds"] == [RESCUE_SEEDS[0]]
    assert plan["next_seed"] == RESCUE_SEEDS[1]


def test_completed_member_with_weakened_evidence_fails_closed(tmp_path: Path) -> None:
    write_member(tmp_path, 0, 41_021, accepted=True)
    terminal_path = next(tmp_path.glob("gpu_*/member_*/worker_terminal.json"))
    terminal = json.loads(terminal_path.read_text())
    terminal["real_division_critical_audit_gate_passed"] = False
    terminal_path.write_text(json.dumps(terminal))

    with pytest.raises(ValueError, match="evidence changed"):
        plan_rescue(tmp_path)


def test_uncommitted_seed_and_duplicate_checkpoint_fail_closed(tmp_path: Path) -> None:
    write_member(tmp_path, 0, 41_021, accepted=True)
    (tmp_path / "gpu_4" / "member_00_seed_99999").mkdir(parents=True)
    with pytest.raises(ValueError, match="uncommitted rescue seeds"):
        plan_rescue(tmp_path)

    unknown = tmp_path / "gpu_4" / "member_00_seed_99999"
    unknown.rmdir()
    write_member(tmp_path, 1, 41_029, accepted=True)
    first_checkpoint = next((tmp_path / "gpu_0").glob("member_*/localization_model.pt"))
    second_checkpoint = next((tmp_path / "gpu_1").glob("member_*/localization_model.pt"))
    second_checkpoint.write_bytes(first_checkpoint.read_bytes())
    second_terminal_path = next((tmp_path / "gpu_1").glob("member_*/worker_terminal.json"))
    second_terminal = json.loads(second_terminal_path.read_text())
    second_terminal["model_sha256"] = hashlib.sha256(second_checkpoint.read_bytes()).hexdigest()
    second_terminal_path.write_text(json.dumps(second_terminal))
    with pytest.raises(ValueError, match="byte-identical"):
        plan_rescue(tmp_path)
