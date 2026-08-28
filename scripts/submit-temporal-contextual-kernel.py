#!/usr/bin/env python
"""Submit only a completed, verified contextual-v3 Kaggle kernel version."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence


COMPETITION = "biohub-cell-tracking-during-development"
RUN_ID = "temporal-contextual-pair-fusion-candidate-v3"
KERNEL_REF = "indarkarhana/biohub-temporal-contextual-submission-candidate-v3"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
CANDIDATE_FAMILY = "trackastra_contextual_pair_fusion_blend"
EXPECTED_BASE_SHA256 = (
    "33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a"
)
EXPECTED_DATASET_SOURCES = [
    "indarkarhana/biohub-temporal-contextual-final-runtime-v1",
    "indarkarhana/biohub-temporal-contextual-exact-acceptance-v3",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
]
EXPECTED_KERNEL_SOURCES = [
    "indarkarhana/biohub-clean-0-927-reproduction-v1",
    "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
    "indarkarhana/biohub-temporal-contextual-transfer-v3",
]
DAILY_SUBMISSION_LIMIT = 5
AUTHORIZATION_SCOPE = (
    "first_hash_bound_candidate_passing_all_external_reciprocal_calibration_"
    "exact_processed_integrity_and_non_replica_gates"
)


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


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True
    )


def parse_json_output(completed: subprocess.CompletedProcess[str], label: str) -> dict:
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} did not emit one JSON object") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} did not emit a JSON object")
    return payload


def validate_kernel_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not (
        payload.get("id") == KERNEL_REF
        and payload.get("is_private") is True
        and payload.get("enable_gpu") is True
        and payload.get("enable_tpu") is False
        and payload.get("enable_internet") is False
        and payload.get("machine_shape") == "NvidiaTeslaT4"
        and payload.get("dataset_sources") == EXPECTED_DATASET_SOURCES
        and payload.get("kernel_sources") == EXPECTED_KERNEL_SOURCES
        and payload.get("competition_sources") == [COMPETITION]
    ):
        raise RuntimeError("contextual candidate kernel metadata is invalid")
    return payload


def validate_submission_authorization(events_path: Path) -> dict:
    authorizations = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        payload = event.get("payload", {})
        fields = payload.get("replacement_fields", {})
        authorization = fields.get("submission_authorization", {})
        if (
            event.get("event_type") == "amendment"
            and event.get("run_id") == RUN_ID
            and fields.get("authorized_for_submission") is True
            and authorization.get("scope") == AUTHORIZATION_SCOPE
            and authorization.get("upload_without_additional_prompt") is True
            and authorization.get("rejected_or_ungated_candidate_upload_forbidden")
            is True
        ):
            authorizations.append(event)
    if len(authorizations) != 1:
        raise RuntimeError("prospective contextual submission authorization is invalid")
    return authorizations[0]


def parse_kernel_status(output: str) -> str:
    match = re.search(
        r'has status\s+"(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', output
    )
    if match is None:
        raise RuntimeError("Kaggle kernel status output is ambiguous")
    return match.group(1).upper()


def validate_remote_kernel_state(payload: dict, version: int) -> None:
    if not (
        payload.get("schema_version") == "biohub.kaggle-kernel-state.v1"
        and payload.get("kernel_slug") == KERNEL_REF
        and payload.get("present") is True
        and payload.get("current_version_number") == version
        and payload.get("is_private") is True
        and payload.get("enable_gpu") is True
        and payload.get("enable_tpu") is False
        and payload.get("enable_internet") is False
        and payload.get("kernel_type") == "notebook"
        and payload.get("dataset_sources") == EXPECTED_DATASET_SOURCES
        and payload.get("kernel_sources") == EXPECTED_KERNEL_SOURCES
        and payload.get("competition_sources") == [COMPETITION]
    ):
        raise RuntimeError("remote contextual candidate kernel state is invalid")


def unique_file_with_hash(root: Path, name: str, digest: str) -> Path:
    matches = [
        path
        for path in root.rglob(name)
        if path.is_file() and sha256_file(path) == digest
    ]
    if not matches:
        raise RuntimeError(f"downloaded kernel output omits hash-bound {name}")
    return sorted(matches, key=lambda path: (len(path.parts), path.as_posix()))[0]


def validate_downloaded_output(root: Path) -> dict[str, Any]:
    terminals = list(root.rglob("candidate_launcher_terminal.json"))
    if len(terminals) != 1:
        raise RuntimeError("downloaded kernel output has ambiguous launcher evidence")
    terminal_path = terminals[0]
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == RUN_ID
        and terminal.get("gpu_count_required") == 2
        and terminal.get("candidate_exists") is True
        and terminal.get("candidate_report_exists") is True
        and terminal.get("root_candidate_exists") is True
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("competition_submission_performed") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise RuntimeError("downloaded candidate launcher evidence is invalid")
    candidate_digest = str(terminal.get("candidate_sha256", ""))
    report_digest = str(terminal.get("report_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_digest) or not re.fullmatch(
        r"[0-9a-f]{64}", report_digest
    ):
        raise RuntimeError("downloaded candidate launcher hashes are invalid")
    candidate = unique_file_with_hash(root, "submission.csv", candidate_digest)
    report_path = unique_file_with_hash(root, "candidate_report.json", report_digest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    coverage = report.get("whole_movie_coverage", [])
    if not (
        report.get("schema_version") == 1
        and report.get("status") == "completed"
        and report.get("candidate_family") == CANDIDATE_FAMILY
        and report.get("appearance_family") == APPEARANCE_FAMILY
        and report.get("gpu_count") == 2
        and report.get("transition_partitioned_inference") is True
        and report.get("transition_partition_kind")
        in {"dominant_movie_transition_split_v1", "whole_movie_lpt_v1"}
        and re.fullmatch(
            r"[0-9a-f]{64}", str(report.get("transition_work_plan_sha256", ""))
        )
        and isinstance(coverage, list)
        and len(coverage) > 0
        and len(coverage) == len(set(coverage))
        and report.get("base_submission_sha256") == EXPECTED_BASE_SHA256
        and report.get("candidate_submission_sha256") == candidate_digest
        and report.get("candidate_submission_sha256") != EXPECTED_BASE_SHA256
        and int(report.get("total_changed_edges", 0)) > 0
        and report.get("nodes_preserved_exactly") is True
        and report.get("public_leaderboard_used_for_selection") is False
        and report.get("competition_submission_performed") is False
    ):
        raise RuntimeError("downloaded contextual candidate report is invalid")
    return {
        "candidate": candidate,
        "candidate_sha256": candidate_digest,
        "candidate_report": report_path,
        "candidate_report_sha256": report_digest,
        "launcher_terminal": terminal_path,
        "launcher_terminal_sha256": sha256_file(terminal_path),
        "total_changed_edges": int(report["total_changed_edges"]),
    }


def submission_command(version: int, message: str) -> list[str]:
    return [
        "kaggle",
        "competitions",
        "submit",
        COMPETITION,
        "--kernel",
        KERNEL_REF,
        "--version",
        str(version),
        "--message",
        message,
    ]


def submission_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def current_daily_submission_count() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:  # pragma: no cover - operator environment
        raise RuntimeError("Kaggle Python SDK is unavailable") from error
    api = KaggleApi()
    api.authenticate()
    today = datetime.now(timezone.utc).date()
    rows = api.competition_submissions(COMPETITION)
    count = 0
    for row in rows:
        observed = submission_date(getattr(row, "date", None))
        if observed is not None and observed.astimezone(timezone.utc).date() == today:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=Path("experiments/events.jsonl"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--message",
        default="Contextual v3: clean reciprocal exact-gated non-replica candidate",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.kernel_version < 1:
        raise ValueError("kernel version must be positive")
    kernel_dir = args.kernel_dir.expanduser().resolve()
    validate_kernel_metadata(kernel_dir / "kernel-metadata.json")
    authorization = validate_submission_authorization(args.events.expanduser().resolve())
    receipt = args.receipt.expanduser().resolve()
    if receipt.exists():
        raise FileExistsError(f"submission receipt already exists: {receipt}")
    download_dir = args.download_dir.expanduser().resolve()
    if download_dir.exists() and (
        not download_dir.is_dir() or any(download_dir.iterdir())
    ):
        raise FileExistsError(f"kernel output directory is not empty: {download_dir}")
    download_dir.mkdir(parents=True, exist_ok=True)

    state = parse_json_output(
        run(
            [
                sys.executable,
                str(Path(__file__).with_name("get-kaggle-kernel-state.py")),
                "--kernel-slug",
                KERNEL_REF,
            ]
        ),
        "owned kernel state",
    )
    validate_remote_kernel_state(state, args.kernel_version)
    status = parse_kernel_status(run(["kaggle", "kernels", "status", KERNEL_REF]).stdout)
    if status != "COMPLETE":
        raise RuntimeError(f"contextual candidate kernel is not complete: {status}")
    run(
        [
            "kaggle",
            "kernels",
            "output",
            KERNEL_REF,
            "--path",
            str(download_dir),
            "--force",
        ]
    )
    output = validate_downloaded_output(download_dir)
    eligible = {
        "schema_version": 1,
        "status": "eligible",
        "run_id": RUN_ID,
        "competition": COMPETITION,
        "kernel_ref": KERNEL_REF,
        "kernel_version": args.kernel_version,
        "candidate_sha256": output["candidate_sha256"],
        "candidate_report_sha256": output["candidate_report_sha256"],
        "launcher_terminal_sha256": output["launcher_terminal_sha256"],
        "total_changed_edges": output["total_changed_edges"],
        "authorization_event_id": authorization["event_id"],
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }
    if not args.execute:
        print(json.dumps(eligible, indent=2, sort_keys=True))
        return

    daily_count = current_daily_submission_count()
    if daily_count >= DAILY_SUBMISSION_LIMIT:
        raise RuntimeError(
            f"daily submission limit reached: {daily_count}/{DAILY_SUBMISSION_LIMIT}"
        )
    completed = run(submission_command(args.kernel_version, args.message))
    atomic_json(
        receipt,
        {
            **eligible,
            "status": "submitted",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "daily_submission_count_before": daily_count,
            "competition_submission_performed": True,
            "message": args.message,
            "kaggle_cli_stdout": completed.stdout.strip(),
        },
    )
    print(receipt)


if __name__ == "__main__":
    main()
