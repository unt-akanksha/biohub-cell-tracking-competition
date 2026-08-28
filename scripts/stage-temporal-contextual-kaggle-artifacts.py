#!/usr/bin/env python
"""Stage accepted contextual-v3 cloud artifacts as two private Kaggle datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


TRANSFER_RUN_ID = "temporal-contextual-pair-fusion-v3"
EXACT_RUN_ID = "trackastra-dual-fold-processed-exact-v1"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
CANDIDATE_FAMILY = "trackastra_contextual_pair_fusion_blend"
APPEARANCE_DATASET_ID = (
    "indarkarhana/biohub-temporal-contextual-transfer-output-v3"
)
ACCEPTANCE_DATASET_ID = (
    "indarkarhana/biohub-temporal-contextual-exact-acceptance-v3"
)
ACCEPTANCE_FILENAME = (
    "temporal-contextual-pair-fusion-v3-exact-acceptance.json"
)
FOLDS = {"target_44b6", "target_6bba"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {resolved}")
    return resolved


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file is missing: {resolved}")
    return resolved


def require_new_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"{label} staging path already exists: {resolved}")
    return resolved


def file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }


def validate_appearance(appearance_root: Path) -> dict:
    terminal_path = appearance_root / "training_terminal.json"
    launcher_path = appearance_root / "cloud_launcher_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    folds = terminal.get("folds", {})
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == TRANSFER_RUN_ID
        and terminal.get("appearance_family") == APPEARANCE_FAMILY
        and terminal.get("gpu_count") == 2
        and terminal.get("both_folds_trained") is True
        and terminal.get("both_folds_improved") is True
        and isinstance(folds, dict)
        and set(folds) == FOLDS
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and launcher.get("schema_version") == 1
        and launcher.get("status") == "completed"
        and launcher.get("run_id") == TRANSFER_RUN_ID
        and launcher.get("gpu_count") == 2
        and launcher.get("training_terminal_sha256") == sha256_file(terminal_path)
        and launcher.get("strict_checkpoint_loaded") is True
        and launcher.get("authorized_for_calibration") is True
        and launcher.get("authorized_for_submission") is False
        and launcher.get("public_predictions_copied") is False
        and launcher.get("public_leaderboard_used_for_selection") is False
        and launcher.get("submission_created") is False
    ):
        raise RuntimeError("contextual-v3 appearance artifact is invalid")
    for fold in FOLDS:
        model = appearance_root / fold / "appearance_model.pt"
        if not model.is_file() or sha256_file(model) != folds[fold].get(
            "model_sha256"
        ):
            raise RuntimeError(f"contextual-v3 appearance model mismatch: {fold}")
    return terminal


def validate_acceptance(payload: dict, appearance: dict) -> None:
    checks = payload.get("gate", {}).get("checks", {})
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "accepted"
        and payload.get("run_id") == EXACT_RUN_ID
        and payload.get("evaluation_kind")
        == "exact_processed_dual_fold_acceptance"
        and payload.get("exact_processed_gate_passed") is True
        and payload.get("candidate_family") == CANDIDATE_FAMILY
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and checks.get("control_score_reproduced") is True
        and checks.get("pooled_score_improved") is True
        and checks.get("per_movie_regression_floor_passed") is True
        and checks.get("node_recall_identical") is True
        and checks.get("edge_sets_differ") is True
        and payload.get("candidate_node_rows_identical") is True
        and payload.get("candidate_edge_sets_differ") is True
        and set(payload.get("appearance_models", {})) == FOLDS
        and payload.get("authorized_for_submission") is False
        and payload.get("competition_submission_performed") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("hyperparameter_selection_performed") is False
    ):
        raise RuntimeError("contextual-v3 exact acceptance artifact is invalid")
    for fold in FOLDS:
        if payload["appearance_models"][fold].get("model_sha256") != appearance[
            "folds"
        ][fold].get("model_sha256"):
            raise RuntimeError(f"accepted appearance model mismatch: {fold}")


def stage_artifacts(
    *,
    appearance_root: Path,
    acceptance_evidence: Path,
    staging_root: Path,
) -> dict[str, Any]:
    appearance_root = require_directory(appearance_root, "appearance")
    acceptance_evidence = require_file(acceptance_evidence, "exact acceptance")
    staging_root = staging_root.expanduser().resolve()
    appearance_target = require_new_directory(
        staging_root / "biohub-temporal-contextual-transfer-output-v3",
        "appearance dataset",
    )
    acceptance_target = require_new_directory(
        staging_root / "biohub-temporal-contextual-exact-acceptance-v3",
        "acceptance dataset",
    )
    appearance = validate_appearance(appearance_root)
    acceptance = json.loads(acceptance_evidence.read_text(encoding="utf-8"))
    validate_acceptance(acceptance, appearance)

    shutil.copytree(appearance_root, appearance_target)
    copied_files = file_manifest(appearance_target)
    atomic_json(
        appearance_target / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": 1,
            "artifact_kind": "contextual_v3_cloud_transfer_output",
            "run_id": TRANSFER_RUN_ID,
            "appearance_family": APPEARANCE_FAMILY,
            "source_training_terminal_sha256": sha256_file(
                appearance_root / "training_terminal.json"
            ),
            "files": copied_files,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )
    atomic_json(
        appearance_target / "dataset-metadata.json",
        {
            "title": "Biohub Temporal Contextual Transfer Output v3",
            "id": APPEARANCE_DATASET_ID,
            "licenses": [{"name": "BSD-3-Clause"}],
            "isPrivate": True,
        },
    )

    acceptance_target.mkdir(parents=True)
    copied_acceptance = acceptance_target / ACCEPTANCE_FILENAME
    shutil.copy2(acceptance_evidence, copied_acceptance)
    atomic_json(
        acceptance_target / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": 1,
            "artifact_kind": "contextual_v3_exact_acceptance",
            "run_id": EXACT_RUN_ID,
            "candidate_family": CANDIDATE_FAMILY,
            "appearance_family": APPEARANCE_FAMILY,
            "acceptance_evidence": {
                "path": ACCEPTANCE_FILENAME,
                "bytes": copied_acceptance.stat().st_size,
                "sha256": sha256_file(copied_acceptance),
            },
            "exact_processed_gate_passed": True,
            "public_leaderboard_used_for_selection": False,
            "competition_submission_performed": False,
        },
    )
    atomic_json(
        acceptance_target / "dataset-metadata.json",
        {
            "title": "Biohub Temporal Contextual Exact Acceptance v3",
            "id": ACCEPTANCE_DATASET_ID,
            "licenses": [{"name": "BSD-3-Clause"}],
            "isPrivate": True,
        },
    )
    return {
        "schema_version": 1,
        "status": "staged",
        "appearance_dataset": str(appearance_target),
        "appearance_dataset_id": APPEARANCE_DATASET_ID,
        "appearance_file_count": len(copied_files),
        "acceptance_dataset": str(acceptance_target),
        "acceptance_dataset_id": ACCEPTANCE_DATASET_ID,
        "acceptance_evidence_sha256": sha256_file(copied_acceptance),
        "kaggle_write_performed": False,
        "competition_submission_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-root", type=Path, required=True)
    parser.add_argument("--acceptance-evidence", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    args = parser.parse_args()
    result = stage_artifacts(
        appearance_root=args.appearance_root,
        acceptance_evidence=args.acceptance_evidence,
        staging_root=args.staging_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
