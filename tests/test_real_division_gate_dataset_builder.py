from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-real-division-gate-dataset.py"


def make_source(tmp_path: Path, module: dict) -> tuple[Path, Path]:
    source = tmp_path / "source"
    hashes = {}
    folds = {}
    for fold, content in (("target_44b6", b"first"), ("target_6bba", b"second")):
        fold_root = source / fold
        fold_root.mkdir(parents=True)
        checkpoint = fold_root / "division_model.pt"
        checkpoint.write_bytes(content)
        digest = module["sha256_file"](checkpoint)
        hashes[fold] = digest
        folds[fold] = {
            "status": "completed",
            "checkpoint_frozen_before_final_probe": True,
            "competition_test_data_read": False,
            "model_sha256": digest,
        }
    terminal = {
        "schema_version": 1,
        "status": "accepted_at_selection",
        "run_id": "competition-real-division-gate-v1",
        "family": "competition_real_temporal_multiscale_division_gate_v1",
        "appearance_family": "temporal_multiscale_contextual_pair_fusion_v4",
        "execution_gpu_count": 1,
        "selection_gate_passed": True,
        "final_probe_opened": False,
        "checkpoint_frozen_before_final_probe": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_final_probe": True,
        "frozen_division_logit_threshold": 1.25,
        "threshold_selection": {"tp": 4, "fp": 0},
        "folds": folds,
    }
    terminal_path = source / "real_division_gate_terminal.json"
    terminal_path.write_text(json.dumps(terminal))
    probe = {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "run_id": "competition-train-focused-division-transfer-probe-v1",
        "model_sha256": [hashes[fold] for fold in ("target_44b6", "target_6bba")],
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "threshold_selected": False,
        "final_probe_opened": True,
        "final_probe_opened_after_training_threshold_freeze": True,
        "submission_created": False,
        "authorized_for_competition_graph_evaluation": True,
        "authorized_for_submission": False,
        "policy_evaluation": {
            "status": "accepted",
            "training_terminal_sha256": module["sha256_file"](terminal_path),
            "model_threshold_frozen_before_probe": 1.25,
            "biological_geometry_minimum": 3.0,
            "conjunctive": {
                "tp": 3,
                "fp": 0,
                "precision": 1.0,
                "jaccard": 1.0,
            },
            "authorized_for_competition_graph_evaluation": True,
            "authorized_for_submission": False,
        },
    }
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe))
    return source, probe_path


def run_builder(module: dict, source: Path, probe: Path, output: Path) -> None:
    previous = sys.argv
    try:
        sys.argv = [
            str(BUILDER),
            "--source-root",
            str(source),
            "--probe-result",
            str(probe),
            "--output-root",
            str(output),
        ]
        module["main"]()
    finally:
        sys.argv = previous


def test_builder_stages_only_accepted_hash_bound_real_gate(tmp_path: Path) -> None:
    module = runpy.run_path(str(BUILDER))
    source, probe = make_source(tmp_path, module)
    output = tmp_path / "dataset"
    run_builder(module, source, probe, output)

    verified = module["verify_dataset"](output)
    metadata = json.loads((output / "dataset-metadata.json").read_text())
    policy = json.loads((output / "division-recovery-policy.json").read_text())
    assert verified["authorized_for_candidate_attachment"] is True
    assert verified["authorized_for_submission"] is False
    assert metadata["id"] == "indarkarhana/biohub-real-division-gate-v1"
    assert policy["biological_geometry_minimum"] == 3.0
    assert policy["public_leaderboard_used_for_selection"] is False


def test_builder_rejects_failed_probe_or_checkpoint_tampering(tmp_path: Path) -> None:
    module = runpy.run_path(str(BUILDER))
    source, probe = make_source(tmp_path, module)
    payload = json.loads(probe.read_text())
    payload["policy_evaluation"]["conjunctive"]["precision"] = 0.5
    probe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source evidence"):
        module["validate_source"](source, probe)

    source, probe = make_source(tmp_path / "again", module)
    (source / "target_44b6/division_model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fold is ineligible"):
        module["validate_source"](source, probe)
