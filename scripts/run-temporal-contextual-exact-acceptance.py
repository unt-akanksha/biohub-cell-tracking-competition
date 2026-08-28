#!/usr/bin/env python
"""Run the pinned contextual-v3 exact processed CPU acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


RUN_ID = "trackastra-dual-fold-processed-exact-v1"
EVALUATION_KIND = "exact_processed_dual_fold_acceptance"
CANDIDATE_FAMILY = "trackastra_contextual_pair_fusion_blend"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
EXPECTED_PROCESSED_CONTROL_SHA256 = (
    "6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b"
)
MAXIMUM_PER_MOVIE_REGRESSION = 0.002


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


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file is missing: {resolved}")
    return resolved


def scoring_command(
    *,
    control_csv: Path,
    candidate_csv: Path,
    truth_dir: Path,
    scorer_lock: Path,
    organizer_checkout: Path,
    tracksdata_checkout: Path,
    materialization_result: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "research.trackastra_graph.score_dual_fold_processed_candidate",
        "--control-csv",
        str(control_csv),
        "--candidate-csv",
        str(candidate_csv),
        "--truth-dir",
        str(truth_dir),
        "--scorer-lock",
        str(scorer_lock),
        "--organizer-checkout",
        str(organizer_checkout),
        "--tracksdata-checkout",
        str(tracksdata_checkout),
        "--materialization-result",
        str(materialization_result),
        "--output",
        str(output),
    ]


def checked_cloud_processed_launcher(
    payload: dict, materialization_result: Path, candidate_csv: Path
) -> None:
    materialization = json.loads(materialization_result.read_text(encoding="utf-8"))
    materialization_valid = bool(
        materialization.get("schema_version") == 1
        and materialization.get("status") == "completed"
        and materialization.get("run_id")
        == "temporal-contextual-pair-fusion-processed-acceptance-v3"
        and materialization.get("evaluation_kind")
        == "predeclared_processed_candidate_materialization"
        and materialization.get("candidate_family") == CANDIDATE_FAMILY
        and materialization.get("appearance_family") == APPEARANCE_FAMILY
        and materialization.get("gpu_count") == 2
        and materialization.get("whole_movie_sharding") is True
        and materialization.get("processed_control_sha256")
        == EXPECTED_PROCESSED_CONTROL_SHA256
        and materialization.get("processed_candidate_sha256")
        == sha256_file(candidate_csv)
        and int(materialization.get("total_changed_edges", 0)) > 0
        and materialization.get("ground_truth_read") is False
        and materialization.get("exact_processed_scoring_performed") is False
        and materialization.get("public_leaderboard_used_for_selection") is False
        and materialization.get("hyperparameter_selection_performed") is False
        and materialization.get("competition_submission_performed") is False
        and materialization.get("authorized_for_submission") is False
    )
    kaggle_launcher = "gpu_count_required" in payload
    launcher_valid = bool(
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_id")
        == "temporal-contextual-pair-fusion-processed-acceptance-v3"
        and payload.get("processed_ground_truth_read") is False
        and payload.get("exact_processed_scoring_performed") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("competition_submission_performed") is False
        and payload.get("authorized_for_submission") is False
        and (
            (
                kaggle_launcher
                and payload.get("gpu_count_required") == 2
                and payload.get("declared_budget_seconds") == 21_600
                and payload.get("materializer_hard_stop_seconds") == 19_800
                and payload.get("whole_movie_sharding_required") is True
                and payload.get("materialization_result_exists") is True
                and payload.get("processed_candidate_exists") is True
                and payload.get("result_sha256") == sha256_file(materialization_result)
                and payload.get("candidate_sha256") == sha256_file(candidate_csv)
            )
            or (
                not kaggle_launcher
                and payload.get("gpu_count") == 2
                and payload.get("processed_control_sha256")
                == EXPECTED_PROCESSED_CONTROL_SHA256
                and payload.get("materialization_result_sha256")
                == sha256_file(materialization_result)
                and payload.get("processed_candidate_sha256")
                == sha256_file(candidate_csv)
                and int(payload.get("total_changed_edges", 0)) > 0
                and payload.get("authorized_for_exact_cpu_scoring") is True
            )
        )
    )
    if not (materialization_valid and launcher_valid):
        raise RuntimeError("contextual-v3 processed launcher evidence is invalid")


def discover_processed_artifacts(
    processed_root: Path,
) -> tuple[Path, Path, Path]:
    materializations = []
    for path in processed_root.rglob("materialization_result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("run_id") == "temporal-contextual-pair-fusion-processed-acceptance-v3":
            materializations.append(path)
    if len(materializations) != 1:
        raise RuntimeError("processed materialization evidence is ambiguous")
    materialization_result = materializations[0]
    candidate_csv = require_file(
        materialization_result.with_name("processed_candidate.csv"),
        "processed candidate",
    )
    launchers = []
    for name in (
        "processed_launcher_terminal.json",
        "cloud_processed_launcher_terminal.json",
    ):
        for path in processed_root.rglob(name):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if payload.get("run_id") == "temporal-contextual-pair-fusion-processed-acceptance-v3":
                launchers.append(path)
    if len(launchers) != 1:
        raise RuntimeError("processed launcher evidence is ambiguous")
    return materialization_result, candidate_csv, launchers[0]


def checked_exact_acceptance(
    payload: dict,
    *,
    control_csv: Path,
    candidate_csv: Path,
    materialization_result: Path,
) -> None:
    checks = payload.get("gate", {}).get("checks", {})
    required_checks = {
        "control_score_reproduced",
        "pooled_score_improved",
        "per_movie_regression_floor_passed",
        "node_recall_identical",
        "edge_sets_differ",
    }
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "accepted"
        and payload.get("run_id") == RUN_ID
        and payload.get("evaluation_kind") == EVALUATION_KIND
        and payload.get("candidate_family") == CANDIDATE_FAMILY
        and payload.get("appearance_family") == APPEARANCE_FAMILY
        and payload.get("exact_processed_gate_passed") is True
        and isinstance(checks, dict)
        and set(checks) == required_checks
        and all(checks.values())
        and float(payload.get("gate", {}).get("worst_movie_score_delta", -1.0))
        >= -MAXIMUM_PER_MOVIE_REGRESSION
        and payload.get("candidate_node_rows_identical") is True
        and payload.get("candidate_edge_sets_differ") is True
        and payload.get("processed_control_sha256") == sha256_file(control_csv)
        and payload.get("processed_candidate_sha256") == sha256_file(candidate_csv)
        and payload.get("materialization_result_sha256")
        == sha256_file(materialization_result)
        and payload.get("authorized_for_submission") is False
        and payload.get("competition_submission_performed") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("hyperparameter_selection_performed") is False
    ):
        raise RuntimeError("contextual-v3 exact CPU acceptance evidence is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--scorer-lock", type=Path, required=True)
    parser.add_argument("--organizer-checkout", type=Path, required=True)
    parser.add_argument("--tracksdata-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    processed_root = require_directory(args.processed_root, "processed")
    control_csv = require_file(args.control_csv, "processed control")
    truth_dir = require_directory(args.truth_dir, "processed truth")
    scorer_lock = require_file(args.scorer_lock, "scorer lock")
    organizer_checkout = require_directory(args.organizer_checkout, "organizer")
    tracksdata_checkout = require_directory(args.tracksdata_checkout, "tracksdata")
    materialization_result, candidate_csv, launcher_path = discover_processed_artifacts(
        processed_root
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exact evidence: {output}")
    if sha256_file(control_csv) != EXPECTED_PROCESSED_CONTROL_SHA256:
        raise RuntimeError("processed control CSV is not the frozen comparator")
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    checked_cloud_processed_launcher(
        launcher, materialization_result, candidate_csv
    )

    subprocess.run(
        scoring_command(
            control_csv=control_csv,
            candidate_csv=candidate_csv,
            truth_dir=truth_dir,
            scorer_lock=scorer_lock,
            organizer_checkout=organizer_checkout,
            tracksdata_checkout=tracksdata_checkout,
            materialization_result=materialization_result,
            output=output,
        ),
        check=True,
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    checked_exact_acceptance(
        evidence,
        control_csv=control_csv,
        candidate_csv=candidate_csv,
        materialization_result=materialization_result,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
