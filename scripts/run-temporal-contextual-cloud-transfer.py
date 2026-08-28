#!/usr/bin/env python
"""Run the frozen contextual-v3 transfer on an exactly two-GPU cloud host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


RUN_ID = "temporal-contextual-pair-fusion-v3"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
)
EXPECTED_TEMPORAL_SOURCE_TREE_SHA256 = (
    "93f98cc5af8fac9bccf9fbe2906228506760916c22100730843e4fe485080e92"
)
SOURCE_HASHES = {
    "research/temporal_contrastive/verify_zebrahub_contextual_acceptance_output.py": (
        "2c9f45effaa8bb5eb23a6d86041cf2811755c1fd8c97ee98223bb5e7d9d2ba96"
    ),
    "research/temporal_contrastive/evaluate_zebrahub_contextual_acceptance.py": (
        "c1ce5bf1f10ba8edc29e0419e80779c6ab9a108a2c224f95b2152c8f6ed46ef2"
    ),
    "research/temporal_contrastive/contextual_pair_fusion.py": (
        "ec3f0e503af039afd879c757a80d805362c36bd66aff44bd588d12c3f5012e7f"
    ),
    "research/temporal_contrastive/train_zebrahub_contextual_pretrain.py": (
        "a2bc71cd8795924e34b54f82a7746de98831ae64c53f77e615791c74b27d095b"
    ),
}


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


def temporal_source_tree_sha256(repository_root: Path) -> tuple[int, str]:
    source_root = repository_root / "research/temporal_contrastive"
    paths = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"temporal source tree is empty: {source_root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return len(paths), digest.hexdigest()


def validate_repository_sources(repository_root: Path) -> dict[str, str]:
    repository_root = require_directory(repository_root, "repository")
    observed: dict[str, str] = {}
    for relative, expected in SOURCE_HASHES.items():
        source = repository_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"cloud verification source is missing: {source}")
        value = sha256_file(source)
        if value != expected:
            raise RuntimeError(f"cloud verification source changed: {relative}: {value}")
        observed[relative] = value
    source_count, source_tree_hash = temporal_source_tree_sha256(repository_root)
    if source_tree_hash != EXPECTED_TEMPORAL_SOURCE_TREE_SHA256:
        raise RuntimeError(
            "temporal source tree changed: "
            f"expected {EXPECTED_TEMPORAL_SOURCE_TREE_SHA256}, saw {source_tree_hash}"
        )
    observed["temporal_source_file_count"] = str(source_count)
    observed["temporal_source_tree_sha256"] = source_tree_hash
    return observed


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


def transfer_command(
    *,
    runtime_root: Path,
    competition_dir: Path,
    synthetic_root: Path,
    pretraining_root: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runtime_root / "train_dual_fold_contextual_pair_fusion.py"),
        "--orchestrate",
        "--competition-dir",
        str(competition_dir),
        "--synthetic-root",
        str(synthetic_root),
        "--initial-model-root",
        str(pretraining_root),
        "--output-dir",
        str(output_dir),
        "--seed",
        "41027",
        "--steps",
        "20000",
        "--base-channels",
        "64",
        "--embedding-channels",
        "256",
        "--real-replay-probability",
        "0.60",
        "--learning-rate",
        "0.00005",
        "--minimum-learning-rate",
        "0.0000005",
        "--ema-decay",
        "0.997",
        "--real-train-movies",
        "96",
        "--real-validation-movies",
        "12",
        "--real-calibration-movies",
        "12",
        "--minimum-real-composite-gain",
        "0.005",
        "--maximum-synthetic-metric-regression",
        "0.01",
        "--validation-every",
        "1000",
        "--max-wall-seconds",
        "36000",
        "--orchestrator-hard-stop-seconds",
        "37800",
        "--finalization-reserve-seconds",
        "1200",
    ]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def checked_acceptance(payload: dict) -> None:
    folds = payload.get("folds")
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "accepted_verified"
        and payload.get("run_id")
        == "zebrahub-contextual-acceptance-evaluation-v1"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and isinstance(folds, dict)
        and set(folds) == {"target_44b6", "target_6bba"}
        and payload.get("competition_data_read") is False
        and payload.get("public_predictions_copied") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ):
        raise RuntimeError("accepted-v3 cloud prerequisite is invalid")


def checked_training(payload: dict) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "verified"
        and payload.get("run_id") == RUN_ID
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("gpu_count") == 2
        and payload.get("strict_checkpoint_loaded") is True
        and payload.get("competition_artifacts_found") is False
        and payload.get("authorized_for_calibration") is True
        and payload.get("authorized_for_submission") is False
    ):
        raise RuntimeError("contextual-v3 cloud transfer output is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--pretraining-root", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repository_root = require_directory(args.repository_root, "repository")
    runtime_root = require_directory(args.runtime_root, "runtime")
    competition_dir = require_directory(args.competition_dir, "competition")
    synthetic_root = require_directory(args.synthetic_root, "synthetic")
    pretraining_root = require_directory(args.pretraining_root, "pretraining")
    acceptance_root = require_directory(args.acceptance_root, "acceptance")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"cloud transfer output is not empty: {output_dir}")

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(f"cloud transfer requires exactly two GPUs, saw {gpu_count}")
    source_hashes = validate_repository_sources(repository_root)
    manifest_hash = sha256_file(runtime_root / "SOURCE_MANIFEST.json")
    if manifest_hash != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError(f"contextual transfer runtime changed: {manifest_hash}")

    started = time.time()
    launcher_terminal = output_dir / "cloud_launcher_terminal.json"
    command = transfer_command(
        runtime_root=runtime_root,
        competition_dir=competition_dir,
        synthetic_root=synthetic_root,
        pretraining_root=pretraining_root,
        output_dir=output_dir,
    )
    base_terminal: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "gpu_count": gpu_count,
        "runtime_manifest_sha256": manifest_hash,
        "source_hashes": source_hashes,
        "competition_submission_command_included": False,
        "public_predictions_copied": False,
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
        acceptance_evidence = run_json(
            [
                sys.executable,
                str(
                    repository_root
                    / "research/temporal_contrastive/verify_zebrahub_contextual_acceptance_output.py"
                ),
                "--output-root",
                str(acceptance_root),
                "--pretraining-root",
                str(pretraining_root),
            ],
            "acceptance verifier",
        )
        checked_acceptance(acceptance_evidence)
        subprocess.run(command, check=True)
        training_terminal = output_dir / "training_terminal.json"
        if not training_terminal.is_file():
            raise RuntimeError("cloud trainer exited without aggregate terminal")
        training_evidence = run_json(
            [
                sys.executable,
                str(runtime_root / "verify_appearance_output.py"),
                "--root",
                str(output_dir),
                "--expected-family",
                APPEARANCE_FAMILY,
                "--strict-checkpoint",
            ],
            "appearance output verifier",
        )
        checked_training(training_evidence)
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "completed",
                "elapsed_seconds": round(time.time() - started, 3),
                "acceptance_terminal_sha256": acceptance_evidence[
                    "acceptance_terminal_sha256"
                ],
                "pretraining_terminal_sha256": acceptance_evidence[
                    "pretraining_terminal_sha256"
                ],
                "training_terminal_sha256": sha256_file(training_terminal),
                "strict_checkpoint_loaded": True,
                "authorized_for_calibration": True,
                "authorized_for_submission": False,
            },
        )
        print(json.dumps(training_evidence, indent=2, sort_keys=True))
    except Exception as error:
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "failed",
                "elapsed_seconds": round(time.time() - started, 3),
                "error": f"{type(error).__name__}: {error}",
                "authorized_for_calibration": False,
                "authorized_for_submission": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
