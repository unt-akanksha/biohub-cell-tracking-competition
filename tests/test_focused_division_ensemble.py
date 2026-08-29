from __future__ import annotations

import json
from pathlib import Path

import pytest

import research.temporal_contrastive.assemble_focused_division_ensemble as ensemble


def make_selected_source(tmp_path: Path) -> tuple[Path, str]:
    fold = "target_44b6"
    fold_root = tmp_path / fold
    fold_root.mkdir(parents=True)
    checkpoint = fold_root / "division_model.pt"
    checkpoint.write_bytes(b"weights")
    digest = ensemble.sha256_file(checkpoint)
    worker = {
        "schema_version": 1,
        "status": "accepted_at_selection",
        "run_id": ensemble.RUN_ID,
        "family": ensemble.FAMILY,
        "appearance_family": ensemble.MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "fold": fold,
        "parameter_count": ensemble.EXPECTED_PARAMETER_COUNT,
        "best_step": 500,
        "selection_gate_passed": True,
        "audit_opened": False,
        "checkpoint_frozen_before_audit": True,
        "model_sha256": digest,
        "initial_selection": {"average_precision": 0.2},
        "best_selection": {"average_precision": 0.3},
        "selection_gate": {"passed": True},
        "initialization": {
            "parent_association_gate_passed": False,
            "initial_predictions_numerically_preserved": True,
            "authorized_for_focused_division_initialization_only": True,
        },
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    config = {"audit_arrays_read": False}
    (fold_root / "worker_terminal.json").write_text(json.dumps(worker))
    (fold_root / "training_config.json").write_text(json.dumps(config))
    return tmp_path, fold


def test_selection_approved_source_can_enter_frozen_ensemble(tmp_path: Path) -> None:
    root, fold = make_selected_source(tmp_path)

    evidence = ensemble.validate_selected_source(root, fold)

    assert evidence["model_sha256"] == ensemble.sha256_file(
        root / fold / "division_model.pt"
    )
    assert evidence["selection_gate"]["passed"] is True


def test_source_is_rejected_after_audit_was_opened(tmp_path: Path) -> None:
    root, fold = make_selected_source(tmp_path)
    worker_path = root / fold / "worker_terminal.json"
    worker = json.loads(worker_path.read_text())
    worker["audit_opened"] = True
    worker_path.write_text(json.dumps(worker))

    with pytest.raises(ValueError, match="source is ineligible"):
        ensemble.validate_selected_source(root, fold)
