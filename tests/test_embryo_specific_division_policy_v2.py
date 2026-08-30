from __future__ import annotations

import hashlib
import json

import pytest

from research.temporal_contrastive.calibrate_embryo_specific_division_policy_v2 import (
    validate_training,
)


def test_calibrator_accepts_frozen_rejected_global_checkpoints(tmp_path) -> None:
    folds = {}
    for fold, content in (("target_44b6", b"a"), ("target_6bba", b"b")):
        path = tmp_path / fold / "division_model.pt"
        path.parent.mkdir()
        path.write_bytes(content)
        folds[fold] = {"model_sha256": hashlib.sha256(content).hexdigest()}
    terminal = {
        "schema_version": 1,
        "status": "rejected_at_selection",
        "run_id": "competition-real-division-hard-negative-gate-v2",
        "family": "competition_real_hard_negative_temporal_division_gate_v2",
        "audit_opened": False,
        "checkpoint_frozen_before_audit": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "folds": folds,
    }
    (tmp_path / "real_division_hard_negative_terminal.json").write_text(
        json.dumps(terminal)
    )

    validated, paths = validate_training(tmp_path)

    assert validated["audit_opened"] is False
    assert len(paths) == 2
    terminal["audit_opened"] = True
    (tmp_path / "real_division_hard_negative_terminal.json").write_text(
        json.dumps(terminal)
    )
    with pytest.raises(ValueError):
        validate_training(tmp_path)
