from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage-temporal-contextual-kaggle-artifacts.py"
SPEC = importlib.util.spec_from_file_location("contextual_artifact_staging", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    appearance = tmp_path / "appearance"
    model_hashes = {}
    for fold, content in (("target_44b6", b"model-a"), ("target_6bba", b"model-b")):
        model = appearance / fold / "appearance_model.pt"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(content)
        model_hashes[fold] = stage.sha256_file(model)
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": stage.TRANSFER_RUN_ID,
        "appearance_family": stage.APPEARANCE_FAMILY,
        "gpu_count": 2,
        "both_folds_trained": True,
        "both_folds_improved": True,
        "folds": {
            fold: {"model_sha256": digest}
            for fold, digest in model_hashes.items()
        },
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    write_json(appearance / "training_terminal.json", terminal)
    write_json(
        appearance / "cloud_launcher_terminal.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": stage.TRANSFER_RUN_ID,
            "gpu_count": 2,
            "training_terminal_sha256": stage.sha256_file(
                appearance / "training_terminal.json"
            ),
            "strict_checkpoint_loaded": True,
            "authorized_for_calibration": True,
            "authorized_for_submission": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )
    acceptance = tmp_path / "acceptance.json"
    write_json(
        acceptance,
        {
            "schema_version": 1,
            "status": "accepted",
            "run_id": stage.EXACT_RUN_ID,
            "evaluation_kind": "exact_processed_dual_fold_acceptance",
            "exact_processed_gate_passed": True,
            "candidate_family": stage.CANDIDATE_FAMILY,
            "appearance_family": stage.APPEARANCE_FAMILY,
            "gate": {
                "checks": {
                    "control_score_reproduced": True,
                    "pooled_score_improved": True,
                    "per_movie_regression_floor_passed": True,
                    "node_recall_identical": True,
                    "edge_sets_differ": True,
                }
            },
            "candidate_node_rows_identical": True,
            "candidate_edge_sets_differ": True,
            "appearance_models": {
                fold: {"model_sha256": digest}
                for fold, digest in model_hashes.items()
            },
            "authorized_for_submission": False,
            "competition_submission_performed": False,
            "public_leaderboard_used_for_selection": False,
            "hyperparameter_selection_performed": False,
        },
    )
    return appearance, acceptance


def test_artifact_staging_is_private_hash_bound_and_write_free(tmp_path: Path) -> None:
    appearance, acceptance = fixture(tmp_path)
    result = stage.stage_artifacts(
        appearance_root=appearance,
        acceptance_evidence=acceptance,
        staging_root=tmp_path / "staging",
    )
    appearance_target = Path(result["appearance_dataset"])
    acceptance_target = Path(result["acceptance_dataset"])
    appearance_metadata = json.loads(
        (appearance_target / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    acceptance_metadata = json.loads(
        (acceptance_target / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    assert appearance_metadata["id"] == stage.APPEARANCE_DATASET_ID
    assert acceptance_metadata["id"] == stage.ACCEPTANCE_DATASET_ID
    assert appearance_metadata["isPrivate"] is True
    assert acceptance_metadata["isPrivate"] is True
    assert result["kaggle_write_performed"] is False
    assert result["competition_submission_performed"] is False
    assert (appearance_target / "target_44b6" / "appearance_model.pt").is_file()
    assert (acceptance_target / stage.ACCEPTANCE_FILENAME).is_file()


def test_artifact_staging_rejects_mutated_accepted_model(tmp_path: Path) -> None:
    appearance, acceptance = fixture(tmp_path)
    payload = json.loads(acceptance.read_text(encoding="utf-8"))
    payload["appearance_models"]["target_44b6"]["model_sha256"] = "0" * 64
    write_json(acceptance, payload)
    with pytest.raises(RuntimeError, match="accepted appearance model mismatch"):
        stage.stage_artifacts(
            appearance_root=appearance,
            acceptance_evidence=acceptance,
            staging_root=tmp_path / "staging",
        )


def test_artifact_staging_refuses_existing_target(tmp_path: Path) -> None:
    appearance, acceptance = fixture(tmp_path)
    target = tmp_path / "staging" / "biohub-temporal-contextual-transfer-output-v3"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        stage.stage_artifacts(
            appearance_root=appearance,
            acceptance_evidence=acceptance,
            staging_root=tmp_path / "staging",
        )


def test_artifact_stager_contains_no_kaggle_write_command() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "kaggle datasets create" not in source
    assert "kaggle datasets version" not in source
    assert "competitions submit" not in source
