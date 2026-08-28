#!/usr/bin/env python
"""Materialize contextual-v3 processed evidence on exactly two cloud GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


RUN_ID = "temporal-contextual-pair-fusion-processed-acceptance-v3"
CALIBRATION_RUN_ID = "temporal-contextual-pair-fusion-blend-v3"
TRANSFER_RUN_ID = "temporal-contextual-pair-fusion-v3"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
CANDIDATE_FAMILY = "trackastra_contextual_pair_fusion_blend"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
)
EXPECTED_PROCESSED_CONTROL_SHA256 = (
    "6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b"
)
EXPECTED_RAW_GRAPH_TREE_SHA256 = (
    "559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e"
)
FOLDS = {"target_44b6", "target_6bba"}
EXPECTED_PROCESSED_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_tree_sha256(root: Path) -> str:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise RuntimeError(f"raw graph tree is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
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


def materialization_command(
    *,
    runtime_root: Path,
    processed_control_csv: Path,
    raw_graph_root: Path,
    competition_dir: Path,
    trackastra_root: Path,
    appearance_root: Path,
    calibration_terminal: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runtime_root / "dual_fold_appearance_processed_acceptance.py"),
        "--orchestrate",
        "--processed-control-csv",
        str(processed_control_csv),
        "--raw-graph-root",
        str(raw_graph_root),
        "--competition-dir",
        str(competition_dir),
        "--trackastra-output-root",
        str(trackastra_root),
        "--appearance-output-root",
        str(appearance_root),
        "--calibration-terminal",
        str(calibration_terminal),
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
        "--hard-stop-seconds",
        "19800",
    ]


def checked_calibration_launcher(
    payload: dict, calibration_root: Path, appearance_root: Path
) -> None:
    calibration_terminal = calibration_root / "calibration_terminal.json"
    transfer_launcher = appearance_root / "cloud_launcher_terminal.json"
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_id") == CALIBRATION_RUN_ID
        and payload.get("gpu_count") == 2
        and payload.get("runtime_manifest_sha256")
        == EXPECTED_RUNTIME_MANIFEST_SHA256
        and payload.get("appearance_training_terminal_sha256")
        == sha256_file(appearance_root / "training_terminal.json")
        and payload.get("transfer_launcher_terminal_sha256")
        == sha256_file(transfer_launcher)
        and payload.get("calibration_terminal_sha256")
        == sha256_file(calibration_terminal)
        and payload.get("both_folds_improved") is True
        and payload.get("authorized_for_processed_materialization") is True
        and payload.get("authorized_for_submission") is False
        and payload.get("processed_acceptance_ground_truth_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ):
        raise RuntimeError("contextual-v3 cloud calibration launcher is invalid")


def checked_calibration(payload: dict) -> None:
    folds = payload.get("folds", {})
    valid_folds = isinstance(folds, dict) and set(folds) == FOLDS
    if valid_folds:
        for row in folds.values():
            selected = row.get("selection", {}).get("selected", {})
            if not (
                row.get("status") == "completed"
                and row.get("appearance_family") == APPEARANCE_FAMILY
                and len(row.get("calibration_stems", [])) == 12
                and row.get("selection", {}).get("improved") is True
                and selected.get("eligible") is True
                and float(selected.get("pooled_gain_vs_zero", float("-inf")))
                >= 0.001
                and float(
                    selected.get("worst_movie_delta_vs_zero", float("-inf"))
                )
                >= -0.002
                and row.get("processed_acceptance_ground_truth_read") is False
                and row.get("public_leaderboard_used_for_selection") is False
                and row.get("submission_created") is False
            ):
                valid_folds = False
                break
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_id") == CALIBRATION_RUN_ID
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and valid_folds
        and payload.get("processed_acceptance_ground_truth_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ):
        raise RuntimeError("contextual-v3 calibration evidence is invalid")


def checked_materialization(
    payload: dict, candidate_csv: Path, calibration_terminal: Path
) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_id") == RUN_ID
        and payload.get("candidate_family") == CANDIDATE_FAMILY
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("evaluation_kind")
        == "predeclared_processed_candidate_materialization"
        and payload.get("gpu_count") == 2
        and payload.get("whole_movie_sharding") is True
        and payload.get("processed_control_sha256")
        == EXPECTED_PROCESSED_CONTROL_SHA256
        and payload.get("processed_candidate_sha256") == sha256_file(candidate_csv)
        and payload.get("processed_candidate_sha256")
        != EXPECTED_PROCESSED_CONTROL_SHA256
        and payload.get("calibration_terminal_sha256")
        == sha256_file(calibration_terminal)
        and set(payload.get("datasets", {})) == EXPECTED_PROCESSED_STEMS
        and set(payload.get("appearance_models", {})) == FOLDS
        and int(payload.get("total_changed_edges", 0)) > 0
        and payload.get("ground_truth_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("hyperparameter_selection_performed") is False
        and payload.get("exact_processed_scoring_performed") is False
        and payload.get("competition_submission_performed") is False
        and payload.get("authorized_for_submission") is False
    ):
        raise RuntimeError("contextual-v3 processed materialization is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--trackastra-root", type=Path, required=True)
    parser.add_argument("--appearance-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--processed-control-csv", type=Path, required=True)
    parser.add_argument("--raw-graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = require_directory(args.runtime_root, "runtime")
    competition_dir = require_directory(args.competition_dir, "competition")
    trackastra_root = require_directory(args.trackastra_root, "Trackastra")
    appearance_root = require_directory(args.appearance_root, "appearance")
    calibration_root = require_directory(args.calibration_root, "calibration")
    raw_graph_root = require_directory(args.raw_graph_root, "raw graph")
    processed_control_csv = args.processed_control_csv.expanduser().resolve()
    if not processed_control_csv.is_file():
        raise FileNotFoundError(
            f"processed control CSV is missing: {processed_control_csv}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"cloud processed output is not empty: {output_dir}")

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(
            f"cloud processed materialization requires exactly two GPUs, saw {gpu_count}"
        )
    manifest_hash = sha256_file(runtime_root / "SOURCE_MANIFEST.json")
    if manifest_hash != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError(f"contextual processed runtime changed: {manifest_hash}")
    control_hash = sha256_file(processed_control_csv)
    if control_hash != EXPECTED_PROCESSED_CONTROL_SHA256:
        raise RuntimeError(f"processed control changed: {control_hash}")
    raw_graph_hash = artifact_tree_sha256(raw_graph_root)
    if raw_graph_hash != EXPECTED_RAW_GRAPH_TREE_SHA256:
        raise RuntimeError(f"processed raw graph tree changed: {raw_graph_hash}")

    calibration_terminal = calibration_root / "calibration_terminal.json"
    calibration = json.loads(calibration_terminal.read_text(encoding="utf-8"))
    checked_calibration(calibration)
    calibration_launcher_path = (
        calibration_root / "cloud_calibration_launcher_terminal.json"
    )
    calibration_launcher = json.loads(
        calibration_launcher_path.read_text(encoding="utf-8")
    )
    checked_calibration_launcher(
        calibration_launcher, calibration_root, appearance_root
    )

    started = time.time()
    launcher_terminal = output_dir / "cloud_processed_launcher_terminal.json"
    base_terminal: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "gpu_count": gpu_count,
        "runtime_manifest_sha256": manifest_hash,
        "processed_control_sha256": control_hash,
        "raw_graph_tree_sha256": raw_graph_hash,
        "calibration_terminal_sha256": sha256_file(calibration_terminal),
        "calibration_launcher_terminal_sha256": sha256_file(
            calibration_launcher_path
        ),
        "processed_ground_truth_read": False,
        "exact_processed_scoring_performed": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
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
        if not (
            appearance_evidence.get("status") == "verified"
            and appearance_evidence.get("run_id") == TRANSFER_RUN_ID
            and appearance_evidence.get("gpu_count") == 2
            and appearance_evidence.get("strict_checkpoint_loaded") is True
            and appearance_evidence.get("authorized_for_submission") is False
        ):
            raise RuntimeError("appearance evidence is invalid")
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
        if not (
            trackastra_evidence.get("status") == "verified"
            and trackastra_evidence.get("gpu_count") == 2
            and set(trackastra_evidence.get("folds", {})) == FOLDS
            and trackastra_evidence.get("authorized_for_submission") is False
        ):
            raise RuntimeError("Trackastra evidence is invalid")
        subprocess.run(
            materialization_command(
                runtime_root=runtime_root,
                processed_control_csv=processed_control_csv,
                raw_graph_root=raw_graph_root,
                competition_dir=competition_dir,
                trackastra_root=trackastra_root,
                appearance_root=appearance_root,
                calibration_terminal=calibration_terminal,
                output_dir=output_dir,
            ),
            check=True,
        )
        result_path = output_dir / "materialization_result.json"
        candidate_csv = output_dir / "processed_candidate.csv"
        if not result_path.is_file() or not candidate_csv.is_file():
            raise RuntimeError("processed materializer omitted required evidence")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checked_materialization(result, candidate_csv, calibration_terminal)
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "completed",
                "elapsed_seconds": round(time.time() - started, 3),
                "materialization_result_sha256": sha256_file(result_path),
                "processed_candidate_sha256": sha256_file(candidate_csv),
                "total_changed_edges": int(result["total_changed_edges"]),
                "authorized_for_exact_cpu_scoring": True,
                "authorized_for_submission": False,
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as error:
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "failed",
                "elapsed_seconds": round(time.time() - started, 3),
                "error": f"{type(error).__name__}: {error}",
                "authorized_for_exact_cpu_scoring": False,
                "authorized_for_submission": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
