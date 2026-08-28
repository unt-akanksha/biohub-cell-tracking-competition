from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit-temporal-contextual-kernel.py"
SPEC = importlib.util.spec_from_file_location("contextual_kernel_submission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_submission_command_targets_exact_code_kernel_version() -> None:
    command = submit.submission_command(7, "clean candidate")
    assert command == [
        "kaggle",
        "competitions",
        "submit",
        submit.COMPETITION,
        "--kernel",
        submit.KERNEL_REF,
        "--version",
        "7",
        "--message",
        "clean candidate",
    ]
    assert "--file" not in command


def test_kernel_metadata_requires_cloud_output_dataset(tmp_path: Path) -> None:
    metadata = tmp_path / "kernel-metadata.json"
    write_json(
        metadata,
        {
            "id": submit.KERNEL_REF,
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": submit.EXPECTED_DATASET_SOURCES,
            "kernel_sources": submit.EXPECTED_KERNEL_SOURCES,
            "competition_sources": [submit.COMPETITION],
        },
    )
    submit.validate_kernel_metadata(metadata)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["dataset_sources"].remove(
        "indarkarhana/biohub-temporal-contextual-transfer-output-v3"
    )
    write_json(metadata, payload)
    with pytest.raises(RuntimeError, match="metadata"):
        submit.validate_kernel_metadata(metadata)


def test_submission_authorization_is_exactly_scoped(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    event = {
        "event_id": "evt-authorized",
        "event_type": "amendment",
        "run_id": submit.RUN_ID,
        "payload": {
            "replacement_fields": {
                "authorized_for_submission": True,
                "submission_authorization": {
                    "scope": submit.AUTHORIZATION_SCOPE,
                    "upload_without_additional_prompt": True,
                    "rejected_or_ungated_candidate_upload_forbidden": True,
                },
            }
        },
    }
    events.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert submit.validate_submission_authorization(events)["event_id"] == (
        "evt-authorized"
    )
    event["payload"]["replacement_fields"]["submission_authorization"][
        "scope"
    ] = "broader"
    events.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="authorization"):
        submit.validate_submission_authorization(events)


def test_downloaded_candidate_is_hash_bound_and_non_replica(tmp_path: Path) -> None:
    candidate = tmp_path / "submission.csv"
    report_path = tmp_path / "temporal_contextual_candidate_v3" / "candidate_report.json"
    candidate.write_bytes(b"candidate")
    candidate_digest = submit.sha256_file(candidate)
    report = {
        "schema_version": 1,
        "status": "completed",
        "candidate_family": submit.CANDIDATE_FAMILY,
        "appearance_family": submit.APPEARANCE_FAMILY,
        "gpu_count": 2,
        "whole_movie_coverage": ["44b6_a", "6bba_b"],
        "base_submission_sha256": submit.EXPECTED_BASE_SHA256,
        "candidate_submission_sha256": candidate_digest,
        "total_changed_edges": 11,
        "nodes_preserved_exactly": True,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }
    write_json(report_path, report)
    report_digest = submit.sha256_file(report_path)
    write_json(
        tmp_path / "candidate_launcher_terminal.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": submit.RUN_ID,
            "gpu_count_required": 2,
            "candidate_exists": True,
            "candidate_report_exists": True,
            "root_candidate_exists": True,
            "candidate_sha256": candidate_digest,
            "report_sha256": report_digest,
            "public_leaderboard_used_for_selection": False,
            "competition_submission_performed": False,
            "authorized_for_submission": False,
        },
    )
    result = submit.validate_downloaded_output(tmp_path)
    assert result["candidate_sha256"] == candidate_digest
    report["candidate_submission_sha256"] = submit.EXPECTED_BASE_SHA256
    write_json(report_path, report)
    with pytest.raises(RuntimeError, match="hash-bound candidate_report.json"):
        submit.validate_downloaded_output(tmp_path)


def test_remote_kernel_must_be_current_private_two_gpu_version() -> None:
    state = {
        "schema_version": "biohub.kaggle-kernel-state.v1",
        "kernel_slug": submit.KERNEL_REF,
        "present": True,
        "current_version_number": 4,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "kernel_type": "notebook",
        "dataset_sources": submit.EXPECTED_DATASET_SOURCES,
        "kernel_sources": submit.EXPECTED_KERNEL_SOURCES,
        "competition_sources": [submit.COMPETITION],
    }
    submit.validate_remote_kernel_state(state, 4)
    state["enable_internet"] = True
    with pytest.raises(RuntimeError, match="remote"):
        submit.validate_remote_kernel_state(state, 4)


def test_kernel_status_parser_is_fail_closed() -> None:
    assert (
        submit.parse_kernel_status(
            'indarkarhana/kernel has status "KernelWorkerStatus.COMPLETE"'
        )
        == "COMPLETE"
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        submit.parse_kernel_status("complete")
