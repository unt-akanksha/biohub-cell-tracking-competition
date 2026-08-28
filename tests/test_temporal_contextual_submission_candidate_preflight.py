from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "build-temporal-contextual-submission-candidate-preflight.py"
)
SPEC = importlib.util.spec_from_file_location("contextual_candidate_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_final_notebook_policy_uses_kaggle_transfer_and_no_upload() -> None:
    code, metadata = preflight.notebook_policy()
    assert metadata["dataset_sources"] == preflight.EXPECTED_DATASET_SOURCES
    assert metadata["kernel_sources"] == preflight.EXPECTED_KERNEL_SOURCES
    assert "torch.cuda.device_count() != 2" in code
    assert "competitions submit" not in code.casefold()


def test_manifest_verification_rejects_mutation(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    source = root / "model.pt"
    source.parent.mkdir()
    source.write_bytes(b"model")
    manifest = {
        "files": {
            "model.pt": {
                "bytes": source.stat().st_size,
                "sha256": preflight.sha256_file(source),
            }
        }
    }
    assert preflight.verify_manifest_files(root, manifest) == [source]
    source.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed"):
        preflight.verify_manifest_files(root, manifest)


def test_kaggle_transfer_and_processed_evidence_are_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    transfer = tmp_path / "transfer"
    training_root = transfer / "temporal_contextual_transfer_v3"
    model_hashes = {}
    folds = {}
    for fold, value in (("target_44b6", b"model-a"), ("target_6bba", b"model-b")):
        model = training_root / fold / "appearance_model.pt"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(value)
        model_hashes[fold] = preflight.sha256_file(model)
        folds[fold] = {
            "status": "completed",
            "appearance_family": preflight.APPEARANCE_FAMILY,
            "best_step": 1000,
            "finetuning_gate_passed": True,
            "model_sha256": model_hashes[fold],
        }
    training = {
        "schema_version": 1,
        "status": "completed",
        "run_id": "temporal-contextual-pair-fusion-v3",
        "appearance_family": preflight.APPEARANCE_FAMILY,
        "gpu_count": 2,
        "both_folds_trained": True,
        "both_folds_improved": True,
        "folds": folds,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    training_terminal = training_root / "training_terminal.json"
    write_json(training_terminal, training)
    write_json(
        transfer / "launcher_terminal.json",
        {
            "schema_version": 1,
            "run_id": "temporal-contextual-pair-fusion-v3",
            "status": "completed",
            "gpu_count_required": 2,
            "declared_budget_seconds": 39_600,
            "trainer_max_wall_seconds": 36_000,
            "trainer_hard_stop_seconds": 37_800,
            "training_terminal_exists": True,
            "training_terminal_sha256": preflight.sha256_file(training_terminal),
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )

    processed = tmp_path / "processed"
    materialization_root = processed / "temporal_contextual_processed_acceptance_v3"
    candidate = materialization_root / "processed_candidate.csv"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"candidate")
    materialization = {
        "schema_version": 1,
        "status": "completed",
        "run_id": "temporal-contextual-pair-fusion-processed-acceptance-v3",
        "candidate_family": preflight.CANDIDATE_FAMILY,
        "appearance_family": preflight.APPEARANCE_FAMILY,
        "evaluation_kind": "predeclared_processed_candidate_materialization",
        "gpu_count": 2,
        "whole_movie_sharding": True,
        "processed_candidate_sha256": preflight.sha256_file(candidate),
        "appearance_models": {
            fold: {"model_sha256": digest}
            for fold, digest in model_hashes.items()
        },
        "datasets": {stem: {} for stem in preflight.EXPECTED_PROCESSED_STEMS},
        "total_changed_edges": 8,
        "ground_truth_read": False,
        "hyperparameter_selection_performed": False,
        "exact_processed_scoring_performed": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    materialization_path = materialization_root / "materialization_result.json"
    write_json(materialization_path, materialization)
    write_json(
        processed / "processed_launcher_terminal.json",
        {
            "schema_version": 1,
            "run_id": "temporal-contextual-pair-fusion-processed-acceptance-v3",
            "status": "completed",
            "declared_budget_seconds": 21_600,
            "materializer_hard_stop_seconds": 19_800,
            "gpu_count_required": 2,
            "whole_movie_sharding_required": True,
            "materialization_result_exists": True,
            "processed_candidate_exists": True,
            "result_sha256": preflight.sha256_file(materialization_path),
            "candidate_sha256": preflight.sha256_file(candidate),
            "processed_ground_truth_read": False,
            "exact_processed_scoring_performed": False,
            "public_leaderboard_used_for_selection": False,
            "competition_submission_performed": False,
            "authorized_for_submission": False,
        },
    )

    acceptance_root = tmp_path / "acceptance"
    acceptance_path = acceptance_root / preflight.ACCEPTANCE_FILENAME
    acceptance = {
        "schema_version": 1,
        "status": "accepted",
        "evaluation_kind": "exact_processed_dual_fold_acceptance",
        "exact_processed_gate_passed": True,
        "candidate_family": preflight.CANDIDATE_FAMILY,
        "appearance_family": preflight.APPEARANCE_FAMILY,
        "candidate_node_rows_identical": True,
        "candidate_edge_sets_differ": True,
        "appearance_models": {
            fold: {"model_sha256": digest}
            for fold, digest in model_hashes.items()
        },
        "processed_candidate_sha256": preflight.sha256_file(candidate),
        "materialization_result_sha256": preflight.sha256_file(
            materialization_path
        ),
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }
    write_json(acceptance_path, acceptance)
    write_json(
        acceptance_root / "dataset-metadata.json",
        {"id": preflight.ACCEPTANCE_DATASET_ID, "isPrivate": True},
    )
    acceptance_hash = preflight.sha256_file(acceptance_path)
    write_json(
        acceptance_root / "ARTIFACT_MANIFEST.json",
        {
            "artifact_kind": "contextual_v3_exact_acceptance",
            "acceptance_evidence": {"sha256": acceptance_hash},
        },
    )

    verified = preflight.verify_staged_inputs(
        transfer, acceptance_root, processed
    )
    assert verified["training_terminal"] == training_terminal
    assert verified["materialization_result"] == materialization_path
    assert verified["processed_candidate"] == candidate


def test_preflight_source_declares_every_guarded_launch_check() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "imports",
        "inputs",
        "single_batch",
        "model_step",
        "checkpoint_roundtrip",
        "output_location",
        "dense_memory",
        "dataset_coverage",
        "non_replica_provenance",
        "quota_policy",
    ):
        assert f'"{name}"' in source
    assert "--strict-checkpoint" in source
    assert "processed_launcher_terminal.json" in source
    assert "materialization_result_sha256" in source
    assert "--processed-root" in source
    assert "competitions submit" not in preflight.notebook_policy()[0].casefold()
