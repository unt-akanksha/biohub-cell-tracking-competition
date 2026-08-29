from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-focused-division-gate-dataset.py"


def make_source(tmp_path: Path, module: dict) -> Path:
    source = tmp_path / "source"
    hashes = {}
    workers = {}
    for fold, payload in (("target_44b6", b"first"), ("target_6bba", b"second")):
        fold_root = source / fold
        fold_root.mkdir(parents=True)
        checkpoint = fold_root / "division_model.pt"
        checkpoint.write_bytes(payload)
        digest = module["sha256_file"](checkpoint)
        hashes[fold] = digest
        workers[fold] = {
            "status": "accepted_at_selection",
            "selection_gate_passed": True,
            "checkpoint_frozen_before_audit": True,
            "model_sha256": digest,
        }
    policy = {
        "schema_version": 1,
        "status": "accepted",
        "run_id": "external-division-recovery-policy-v1",
        "model_training_run_id": "focused-division-gate-v1",
        "model_sha256": hashes,
        "audit_opened_after_threshold_freeze": True,
        "authorized_for_competition_graph_evaluation": True,
        "authorized_for_submission": False,
        "competition_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    terminal = {
        "schema_version": 1,
        "status": "accepted",
        "run_id": "focused-division-gate-v1",
        "execution_gpu_count": 1,
        "both_folds_selected": True,
        "audit_opened": True,
        "audit_opened_after_threshold_freeze": True,
        "authorized_for_competition_graph_evaluation": True,
        "authorized_for_submission": False,
        "model_sha256": hashes,
        "folds": workers,
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    (source / "division-recovery-policy.json").write_text(json.dumps(policy))
    (source / "focused_division_gate_terminal.json").write_text(json.dumps(terminal))
    return source


def test_builder_stages_only_hash_bound_focused_gate(tmp_path: Path) -> None:
    module = runpy.run_path(str(BUILDER))
    source = make_source(tmp_path, module)
    output = tmp_path / "dataset"
    previous = sys.argv
    try:
        sys.argv = [
            str(BUILDER),
            "--source-root",
            str(source),
            "--output-root",
            str(output),
        ]
        module["main"]()
    finally:
        sys.argv = previous

    verified = module["verify_dataset"](output)
    metadata = json.loads((output / "dataset-metadata.json").read_text())
    assert verified["authorized_for_candidate_attachment"] is True
    assert verified["authorized_for_submission"] is False
    assert metadata["id"] == "indarkarhana/biohub-focused-division-gate-v1"


def test_dataset_verifier_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    module = runpy.run_path(str(BUILDER))
    source = make_source(tmp_path, module)
    output = tmp_path / "dataset"
    previous = sys.argv
    try:
        sys.argv = [
            str(BUILDER),
            "--source-root",
            str(source),
            "--output-root",
            str(output),
        ]
        module["main"]()
    finally:
        sys.argv = previous
    (output / "target_44b6/division_model.pt").write_bytes(b"changed")

    with pytest.raises(ValueError, match="staged file changed"):
        module["verify_dataset"](output)
