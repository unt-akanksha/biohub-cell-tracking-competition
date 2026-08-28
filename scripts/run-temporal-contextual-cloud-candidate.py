#!/usr/bin/env python
"""Build the accepted contextual-v3 candidate on exactly two cloud GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


RUN_ID = "temporal-contextual-pair-fusion-candidate-v3"
TRANSFER_RUN_ID = "temporal-contextual-pair-fusion-v3"
EXACT_RUN_ID = "trackastra-dual-fold-processed-exact-v1"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
CANDIDATE_FAMILY = "trackastra_contextual_pair_fusion_blend"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
)
EXPECTED_BASE_SHA256 = (
    "33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a"
)
EXPECTED_RAW_GRAPH_TREE_SHA256 = (
    "559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e"
)
FOLDS = {"target_44b6", "target_6bba"}
INFERENCE_HARD_STOP_SECONDS = 36_000
FINALIZATION_RESERVE_SECONDS = 7_200


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
        raise RuntimeError(f"base raw graph tree is empty: {root}")
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


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file is missing: {resolved}")
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


def candidate_command(
    *,
    runtime_root: Path,
    base_submission: Path,
    base_graph_root: Path,
    image_root: Path,
    trackastra_root: Path,
    appearance_root: Path,
    acceptance_evidence: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runtime_root / "dual_fold_appearance_submission.py"),
        "--orchestrate",
        "--base-submission",
        str(base_submission),
        "--base-graph-root",
        str(base_graph_root),
        "--image-root",
        str(image_root),
        "--trackastra-44b6-dir",
        str(trackastra_root / "target_44b6"),
        "--trackastra-6bba-dir",
        str(trackastra_root / "target_6bba"),
        "--appearance-44b6-model",
        str(appearance_root / "target_44b6" / "appearance_model.pt"),
        "--appearance-6bba-model",
        str(appearance_root / "target_6bba" / "appearance_model.pt"),
        "--acceptance-evidence",
        str(acceptance_evidence),
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
        str(INFERENCE_HARD_STOP_SECONDS),
    ]


def checked_exact_acceptance(payload: dict) -> None:
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
        and isinstance(checks, dict)
        and checks.get("control_score_reproduced") is True
        and checks.get("pooled_score_improved") is True
        and checks.get("per_movie_regression_floor_passed") is True
        and checks.get("node_recall_identical") is True
        and checks.get("edge_sets_differ") is True
        and set(payload.get("models", {})) == FOLDS
        and set(payload.get("appearance_models", {})) == FOLDS
        and set(payload.get("appearance_blend", {})) == FOLDS
        and payload.get("candidate_node_rows_identical") is True
        and payload.get("candidate_edge_sets_differ") is True
        and payload.get("authorized_for_submission") is False
        and payload.get("competition_submission_performed") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("hyperparameter_selection_performed") is False
    ):
        raise RuntimeError("contextual-v3 exact acceptance is invalid")


def checked_appearance(payload: dict) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "verified"
        and payload.get("run_id") == TRANSFER_RUN_ID
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("gpu_count") == 2
        and set(payload.get("folds", {})) == FOLDS
        and payload.get("strict_checkpoint_loaded") is True
        and payload.get("competition_artifacts_found") is False
        and payload.get("authorized_for_submission") is False
    ):
        raise RuntimeError("contextual-v3 appearance output is invalid")


def checked_trackastra(payload: dict) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "verified"
        and payload.get("gpu_count") == 2
        and set(payload.get("folds", {})) == FOLDS
        and payload.get("source_policy")
        in {"adapted_dual_fold", "predeclared_pretrained_control"}
        and payload.get("competition_artifacts_found") is False
        and payload.get("authorized_for_submission") is False
    ):
        raise RuntimeError("Trackastra candidate source is invalid")


def checked_candidate(
    payload: dict,
    *,
    candidate_csv: Path,
    acceptance_evidence: Path,
) -> None:
    coverage = payload.get("whole_movie_coverage", [])
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("candidate_family") == CANDIDATE_FAMILY
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("gpu_count") == 2
        and payload.get("inference_hard_stop_seconds")
        == INFERENCE_HARD_STOP_SECONDS
        and payload.get("notebook_runtime_reserve_seconds")
        >= FINALIZATION_RESERVE_SECONDS
        and isinstance(coverage, list)
        and len(coverage) > 0
        and len(coverage) == len(set(coverage))
        and payload.get("base_submission_sha256") == EXPECTED_BASE_SHA256
        and payload.get("candidate_submission_sha256") == sha256_file(candidate_csv)
        and payload.get("candidate_submission_sha256") != EXPECTED_BASE_SHA256
        and payload.get("acceptance_evidence_sha256")
        == sha256_file(acceptance_evidence)
        and set(payload.get("models", {})) == FOLDS
        and set(payload.get("appearance_models", {})) == FOLDS
        and set(payload.get("appearance_blend", {})) == FOLDS
        and int(payload.get("total_changed_edges", 0)) > 0
        and payload.get("nodes_preserved_exactly") is True
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("competition_submission_performed") is False
    ):
        raise RuntimeError("contextual-v3 final candidate is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--trackastra-root", type=Path, required=True)
    parser.add_argument("--appearance-root", type=Path, required=True)
    parser.add_argument("--acceptance-evidence", type=Path, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--base-graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = require_directory(args.runtime_root, "runtime")
    competition_dir = require_directory(args.competition_dir, "competition")
    image_root = require_directory(competition_dir / "test", "competition test")
    trackastra_root = require_directory(args.trackastra_root, "Trackastra")
    appearance_root = require_directory(args.appearance_root, "appearance")
    acceptance_evidence = require_file(args.acceptance_evidence, "exact acceptance")
    base_submission = require_file(args.base_submission, "base submission")
    base_graph_root = require_directory(args.base_graph_root, "base raw graph")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"cloud candidate output is not empty: {output_dir}")

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(
            f"cloud candidate requires exactly two GPUs, saw {gpu_count}"
        )
    manifest_hash = sha256_file(runtime_root / "SOURCE_MANIFEST.json")
    if manifest_hash != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError(f"contextual candidate runtime changed: {manifest_hash}")
    base_hash = sha256_file(base_submission)
    if base_hash != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"candidate base submission changed: {base_hash}")
    graph_hash = artifact_tree_sha256(base_graph_root)
    if graph_hash != EXPECTED_RAW_GRAPH_TREE_SHA256:
        raise RuntimeError(f"candidate base raw graph tree changed: {graph_hash}")
    acceptance = json.loads(acceptance_evidence.read_text(encoding="utf-8"))
    checked_exact_acceptance(acceptance)

    started = time.time()
    launcher_terminal = output_dir / "cloud_candidate_launcher_terminal.json"
    base_terminal: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "gpu_count": gpu_count,
        "runtime_manifest_sha256": manifest_hash,
        "base_submission_sha256": base_hash,
        "base_raw_graph_tree_sha256": graph_hash,
        "acceptance_evidence_sha256": sha256_file(acceptance_evidence),
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
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
            candidate_command(
                runtime_root=runtime_root,
                base_submission=base_submission,
                base_graph_root=base_graph_root,
                image_root=image_root,
                trackastra_root=trackastra_root,
                appearance_root=appearance_root,
                acceptance_evidence=acceptance_evidence,
                output_dir=output_dir,
            ),
            check=True,
        )
        candidate_csv = output_dir / "submission.csv"
        report_path = output_dir / "candidate_report.json"
        if not candidate_csv.is_file() or not report_path.is_file():
            raise RuntimeError("candidate inference omitted required output files")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checked_candidate(
            report,
            candidate_csv=candidate_csv,
            acceptance_evidence=acceptance_evidence,
        )
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "completed",
                "elapsed_seconds": round(time.time() - started, 3),
                "candidate_report_sha256": sha256_file(report_path),
                "candidate_submission_sha256": sha256_file(candidate_csv),
                "total_changed_edges": int(report["total_changed_edges"]),
                "nodes_preserved_exactly": True,
                "ready_for_submission_upload": True,
            },
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as error:
        atomic_json(
            launcher_terminal,
            {
                **base_terminal,
                "status": "failed",
                "elapsed_seconds": round(time.time() - started, 3),
                "error": f"{type(error).__name__}: {error}",
                "ready_for_submission_upload": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
