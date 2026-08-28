#!/usr/bin/env python
"""Run frozen contextual-v3 calibration on an exactly two-GPU cloud host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


RUN_ID = "temporal-contextual-pair-fusion-blend-v3"
TRANSFER_RUN_ID = "temporal-contextual-pair-fusion-v3"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
)
EXPECTED_TEMPORAL_SOURCE_TREE_SHA256 = (
    "9c1db520a098d70c746de2afbb4e1037796731cee9366c5aa0bd4d5f16d2b9c2"
)
FOLDS = {"target_44b6", "target_6bba"}
MINIMUM_POOLED_GAIN = 0.001
MAXIMUM_MOVIE_REGRESSION = 0.002


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {resolved}")
    return resolved


def parse_json_output(completed: subprocess.CompletedProcess[str], label: str) -> dict:
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} did not emit one JSON object") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} did not emit a JSON object")
    return payload


def run_json(command: Sequence[str], label: str) -> dict:
    completed = subprocess.run(
        list(command), check=True, capture_output=True, text=True
    )
    return parse_json_output(completed, label)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def calibration_command(
    *,
    runtime_root: Path,
    appearance_root: Path,
    trackastra_root: Path,
    competition_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runtime_root / "calibrate_dual_fold_blend.py"),
        "--orchestrate",
        "--appearance-output-root",
        str(appearance_root),
        "--trackastra-output-root",
        str(trackastra_root),
        "--competition-dir",
        str(competition_dir),
        "--trackastra-dir",
        str(runtime_root / "trackastra_source"),
        "--output-dir",
        str(output_dir),
        "--max-tokens",
        "512",
        "--candidate-radius",
        "80.0",
        "--node-batch-size",
        "64",
        "--max-wall-seconds",
        "18000",
        "--orchestrator-hard-stop-seconds",
        "19800",
    ]


def checked_transfer_launcher(payload: dict, appearance_root: Path) -> None:
    source_hashes = payload.get("source_hashes", {})
    training_terminal = appearance_root / "training_terminal.json"
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_id") == TRANSFER_RUN_ID
        and payload.get("gpu_count") == 2
        and payload.get("runtime_manifest_sha256")
        == EXPECTED_RUNTIME_MANIFEST_SHA256
        and isinstance(source_hashes, dict)
        and source_hashes.get("temporal_source_tree_sha256")
        == EXPECTED_TEMPORAL_SOURCE_TREE_SHA256
        and payload.get("training_terminal_sha256")
        == sha256_file(training_terminal)
        and payload.get("strict_checkpoint_loaded") is True
        and payload.get("authorized_for_calibration") is True
        and payload.get("authorized_for_submission") is False
        and payload.get("public_predictions_copied") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ):
        raise RuntimeError("contextual-v3 cloud transfer launcher is invalid")


def checked_appearance(payload: dict) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "verified"
        and payload.get("run_id") == TRANSFER_RUN_ID
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("gpu_count") == 2
        and payload.get("strict_checkpoint_loaded") is True
        and payload.get("competition_artifacts_found") is False
        and payload.get("authorized_for_calibration") is True
        and payload.get("authorized_for_submission") is False
    ):
        raise RuntimeError("contextual-v3 appearance output is invalid")


def checked_trackastra(payload: dict) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "verified"
        and payload.get("gpu_count") == 2
        and payload.get("source_policy")
        in {"adapted_dual_fold", "predeclared_pretrained_control"}
        and set(payload.get("folds", {})) == FOLDS
        and payload.get("competition_artifacts_found") is False
        and payload.get("authorized_for_submission") is False
    ):
        raise RuntimeError("Trackastra calibration source is invalid")


def checked_calibration(payload: dict) -> None:
    folds = payload.get("folds", {})
    valid_folds = isinstance(folds, dict) and set(folds) == FOLDS
    if valid_folds:
        for row in folds.values():
            selection = row.get("selection", {})
            selected = selection.get("selected", {})
            grid = selection.get("grid", [])
            has_zero_control = any(
                item.get("ensemble_mode") == "target_only"
                and float(item.get("appearance_weight", -1.0)) == 0.0
                and float(item.get("division_weight", -1.0)) == 0.0
                for item in grid
            )
            if not (
                row.get("schema_version") == 1
                and row.get("status") == "completed"
                and row.get("run_id") == RUN_ID
                and row.get("appearance_family") == APPEARANCE_FAMILY
                and len(row.get("calibration_stems", [])) == 12
                and selection.get("improved") is True
                and selected.get("eligible") is True
                and float(selected.get("pooled_gain_vs_zero", float("-inf")))
                >= MINIMUM_POOLED_GAIN
                and float(
                    selected.get("worst_movie_delta_vs_zero", float("-inf"))
                )
                >= -MAXIMUM_MOVIE_REGRESSION
                and has_zero_control
                and row.get("processed_acceptance_ground_truth_read") is False
                and row.get("public_leaderboard_used_for_selection") is False
                and row.get("submission_created") is False
            ):
                valid_folds = False
                break
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_id") == RUN_ID
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and valid_folds
        and payload.get("processed_acceptance_ground_truth_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ):
        raise RuntimeError("contextual-v3 cloud calibration output is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--trackastra-root", type=Path, required=True)
    parser.add_argument("--appearance-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = require_directory(args.runtime_root, "runtime")
    competition_dir = require_directory(args.competition_dir, "competition")
    trackastra_root = require_directory(args.trackastra_root, "Trackastra")
    appearance_root = require_directory(args.appearance_root, "appearance")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"cloud calibration output is not empty: {output_dir}")

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(
            f"cloud calibration requires exactly two GPUs, saw {gpu_count}"
        )
    manifest_hash = sha256_file(runtime_root / "SOURCE_MANIFEST.json")
    if manifest_hash != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError(f"contextual calibration runtime changed: {manifest_hash}")
    transfer_launcher_path = appearance_root / "cloud_launcher_terminal.json"
    transfer_launcher = json.loads(transfer_launcher_path.read_text(encoding="utf-8"))
    checked_transfer_launcher(transfer_launcher, appearance_root)

    started = time.time()
    launcher_terminal = output_dir / "cloud_calibration_launcher_terminal.json"
    base_terminal: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "gpu_count": gpu_count,
        "runtime_manifest_sha256": manifest_hash,
        "appearance_training_terminal_sha256": sha256_file(
            appearance_root / "training_terminal.json"
        ),
        "transfer_launcher_terminal_sha256": sha256_file(transfer_launcher_path),
        "processed_acceptance_ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    try:
        runtime_evidence = run_json(
            [
                sys.executable,
                str(runtime_root / "verify_runtime.py"),
                "--root",
                str(runtime_root),
                "--require-gpus",
            ],
            "runtime verifier",
        )
        if not (
            runtime_evidence.get("status") == "verified"
            and runtime_evidence.get("manifest_sha256") == manifest_hash
            and runtime_evidence.get("required_gpu_count") == 2
            and runtime_evidence.get("detected_gpu_count") == 2
            and runtime_evidence.get("submission_command_included") is False
        ):
            raise RuntimeError("runtime verification evidence is invalid")
        appearance_evidence = run_json(
            [
                sys.executable,
                str(runtime_root / "verify_appearance_output.py"),
                "--root",
                str(appearance_root),
                "--expected-family",
                APPEARANCE_FAMILY,
                "--strict-checkpoint",
            ],
            "appearance output verifier",
        )
        checked_appearance(appearance_evidence)
        trackastra_evidence = run_json(
            [
                sys.executable,
                str(runtime_root / "verify_trackastra_output.py"),
                "--root",
                str(trackastra_root),
                "--allow-pretrained-control",
            ],
            "Trackastra output verifier",
        )
        checked_trackastra(trackastra_evidence)
        subprocess.run(
            calibration_command(
                runtime_root=runtime_root,
                appearance_root=appearance_root,
                trackastra_root=trackastra_root,
                competition_dir=competition_dir,
                output_dir=output_dir,
            ),
            check=True,
        )
        calibration_terminal = output_dir / "calibration_terminal.json"
        if not calibration_terminal.is_file():
            raise RuntimeError("cloud calibrator exited without aggregate terminal")
        calibration = json.loads(calibration_terminal.read_text(encoding="utf-8"))
        checked_calibration(calibration)
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "completed",
                "elapsed_seconds": round(time.time() - started, 3),
                "calibration_terminal_sha256": sha256_file(calibration_terminal),
                "both_folds_improved": True,
                "authorized_for_processed_materialization": True,
                "authorized_for_submission": False,
            },
        )
        print(json.dumps(calibration, indent=2, sort_keys=True))
    except Exception as error:
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "failed",
                "elapsed_seconds": round(time.time() - started, 3),
                "error": f"{type(error).__name__}: {error}",
                "authorized_for_processed_materialization": False,
                "authorized_for_submission": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
