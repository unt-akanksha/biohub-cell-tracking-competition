from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.trackastra_graph.verify_dual_fold_training_output import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_SYNTHETIC_MANIFEST_SHA256,
    FOLDS,
    RUN_ID,
    verify_output,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_output(tmp_path: Path) -> Path:
    aggregate_folds = {}
    for index, (fold, prefixes) in enumerate(FOLDS.items(), start=1):
        fold_dir = tmp_path / fold
        fold_dir.mkdir(parents=True)
        model = fold_dir / "model.pt"
        model.write_bytes(f"model-{fold}".encode())
        (fold_dir / "config.yaml").write_text("model: fixture\n", encoding="utf-8")
        worker = {
            "schema_version": 1,
            "status": "completed",
            "run_id": RUN_ID,
            "fold": fold,
            "completed_step": 3000,
            "best_step": index * 500,
            "pretrained_initialization_retained": False,
            "initial_selection_score": 0.70,
            "best_selection_score": 0.72,
            "initial_real": {"composite": 0.70},
            "best_real": {"composite": 0.705},
            "initial_synthetic": {"composite": 0.80},
            "best_synthetic": {"composite": 0.799},
            "model_sha256": digest(model),
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "submission_created": False,
        }
        config = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "fold": fold,
            "source_prefix": prefixes["source_prefix"],
            "target_prefix": prefixes["target_prefix"],
            "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
            "synthetic_manifest_sha256": EXPECTED_SYNTHETIC_MANIFEST_SHA256,
            "synthetic_native_geometry_restored": True,
            "synthetic_train_count": 2046,
            "synthetic_validation_count": 128,
            "steps_target": 75000,
            "real_train_stems": [
                f"{prefixes['source_prefix']}_train_{row:03d}" for row in range(96)
            ],
            "real_validation_stems": [
                f"{prefixes['target_prefix']}_valid_{row:03d}" for row in range(12)
            ],
        }
        write_json(fold_dir / "worker_terminal.json", worker)
        write_json(fold_dir / "training_config.json", config)
        aggregate_folds[fold] = worker
    write_json(
        tmp_path / "training_terminal.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": RUN_ID,
            "gpu_count": 2,
            "whole_fold_coverage": sorted(FOLDS),
            "folds": aggregate_folds,
            "both_folds_improved": True,
            "submission_created": False,
        },
    )
    return tmp_path


def test_verifier_binds_both_improved_models_and_clean_inventories(
    tmp_path: Path,
) -> None:
    result = verify_output(valid_output(tmp_path))

    assert result["status"] == "verified"
    assert set(result["folds"]) == set(FOLDS)
    assert result["competition_artifacts_found"] is False
    assert result["authorized_for_submission"] is False


def test_verifier_rejects_worker_aggregate_divergence(tmp_path: Path) -> None:
    root = valid_output(tmp_path)
    aggregate = json.loads((root / "training_terminal.json").read_text())
    aggregate["folds"]["target_44b6"]["best_step"] = 2500
    write_json(root / "training_terminal.json", aggregate)

    with pytest.raises(ValueError, match="diverge"):
        verify_output(root)


def test_verifier_rejects_gain_failure_and_competition_artifact(
    tmp_path: Path,
) -> None:
    root = valid_output(tmp_path)
    worker_path = root / "target_6bba" / "worker_terminal.json"
    worker = json.loads(worker_path.read_text())
    worker["best_real"]["composite"] = worker["initial_real"]["composite"]
    write_json(worker_path, worker)
    aggregate_path = root / "training_terminal.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["folds"]["target_6bba"] = deepcopy(worker)
    write_json(aggregate_path, aggregate)

    with pytest.raises(ValueError, match="real reciprocal gain"):
        verify_output(root)

    worker["best_real"]["composite"] = 0.705
    write_json(worker_path, worker)
    aggregate["folds"]["target_6bba"] = deepcopy(worker)
    write_json(aggregate_path, aggregate)
    (root / "submission.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="competition artifacts"):
        verify_output(root)
