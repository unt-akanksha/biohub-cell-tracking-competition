from __future__ import annotations

import json
from pathlib import Path

from research.score_handcrafted_division_probe import validate_model_and_policy
from research.train_handcrafted_division_gate import sha256_file


def test_handcrafted_probe_requires_frozen_unopened_policy(tmp_path: Path) -> None:
    root = tmp_path / "training"
    root.mkdir()
    model = root / "handcrafted_division_gate.joblib"
    model.write_bytes(b"model")
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": "competition-real-handcrafted-division-gate-v1",
        "feature_family": "temporal_radial_peak_morphology_v1",
        "final_probe_opened": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "model_sha256": sha256_file(model),
    }
    terminal_path = root / "handcrafted_division_gate_terminal.json"
    terminal_path.write_text(json.dumps(terminal))
    policy = {
        "schema_version": 1,
        "status": "accepted_at_selection",
        "run_id": "competition-real-handcrafted-division-policy-v1",
        "model_training_run_id": "competition-real-handcrafted-division-gate-v1",
        "model_sha256": sha256_file(model),
        "training_terminal_sha256": sha256_file(terminal_path),
        "selection_gate_passed": True,
        "final_probe_opened": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_final_probe": True,
        "authorized_for_submission": False,
        "frozen_division_probability_threshold": 0.9,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    _, validated, path = validate_model_and_policy(root, policy_path)

    assert validated["frozen_division_probability_threshold"] == 0.9
    assert path == model
