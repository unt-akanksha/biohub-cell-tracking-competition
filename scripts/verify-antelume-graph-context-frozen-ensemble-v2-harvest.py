#!/usr/bin/env python3
"""Verify and extract the frozen graph-context ensemble result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


ROOT = "competition-graph-context-division-frozen-ensemble-v2"
TRAIN_RUN_ID = "competition-graph-context-division-frozen-ensemble-v2"
PROBE_RUN_ID = "competition-graph-context-division-development-probe-v1"
POLICY = "equal_rank_selection_admitted_ensemble"
POLICY_CONTRACT = "all-selection-admitted-equal-rank-ensemble-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != ROOT:
        raise ValueError(f"unsafe graph-context archive member: {name}")
    return path


def metric_gate(metrics: dict) -> bool:
    by_embryo = metrics.get("by_embryo", {})
    return bool(
        metrics.get("average_precision", 0.0) >= 0.55
        and metrics.get("true_positives_before_first_false_positive", 0) >= 2
        and len(by_embryo) == 2
        and all(row.get("average_precision", 0.0) >= 0.40 for row in by_embryo.values())
    )


def verify_tree(root: Path) -> dict:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ValueError("graph-context v2 SHA256SUMS is missing")
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        safe_name(relative)
        if relative in expected:
            raise ValueError("duplicate graph-context v2 hash entry")
        expected[relative] = digest
    observed = {
        path.relative_to(root.parent).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(expected) != set(observed):
        raise ValueError("graph-context v2 archive files are not exactly hash-bound")
    for relative, path in observed.items():
        if sha256_file(path) != expected[relative]:
            raise ValueError(f"graph-context v2 file hash changed: {relative}")

    training_exit = int((root / "training.exit-code").read_text().strip())
    probe_exit = int((root / "development-probe.exit-code").read_text().strip())
    terminal_path = root / "models/graph_context_division_sweep_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    members = [str(value) for value in terminal.get("deployment_members", [])]
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == TRAIN_RUN_ID
        and terminal.get("planned_model_count") == 8
        and terminal.get("completed_model_count") == 8
        and terminal.get("steps_per_model") == 20_000
        and terminal.get("policy_contract") == POLICY_CONTRACT
        and terminal.get("constituent_audit_gate_required") is False
        and terminal.get("ensemble_members_precommitted_before_audit") is True
        and terminal.get("model_subset_searched_on_audit") is False
        and terminal.get("absolute_threshold_used_for_deployment") is False
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise ValueError("graph-context v2 aggregate contract changed")

    probe = None
    if training_exit == 0:
        probe_path = root / "graph_context_development_probe.json"
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        if not (
            terminal.get("policy_audit_passed") is True
            and terminal.get("policy_unit_audited") is True
            and terminal.get("deployment_policy") == POLICY
            and terminal.get("precommitted_policy") == POLICY
            and members == terminal.get("precommitted_members")
            and 2 <= len(members) <= 8
            and len(members) == len(set(members))
            and terminal.get("ensemble_eligible") is True
            and metric_gate(terminal.get("audit_ensemble") or {})
            and probe_exit == 0
            and probe.get("schema_version") == 1
            and probe.get("status") == "development_probe_complete"
            and probe.get("run_id") == PROBE_RUN_ID
            and probe.get("training_run_id") == TRAIN_RUN_ID
            and probe.get("policy_contract") == POLICY_CONTRACT
            and probe.get("selection_policy") == POLICY
            and probe.get("member_count") == len(members)
            and probe.get("absolute_threshold_used") is False
            and probe.get("weights_searched_on_probe") is False
            and probe.get("model_subset_searched_on_probe") is False
            and probe.get("competition_test_data_read") is False
            and probe.get("public_leaderboard_used_for_selection") is False
            and probe.get("authorized_for_submission") is False
        ):
            raise ValueError("accepted graph-context v2 probe contract changed")
        for member in members:
            worker = json.loads((root / "models" / member / "worker_terminal.json").read_text())
            audit = json.loads((root / "models" / member / "audit_terminal.json").read_text())
            checkpoint = root / "models" / member / "graph_context_model.pt"
            if not (
                worker.get("run_id") == TRAIN_RUN_ID
                and worker.get("status") == "accepted_at_selection"
                and worker.get("selection_gate_passed") is True
                and worker.get("model_sha256") == sha256_file(checkpoint)
                and audit.get("member") == member
                and audit.get("model_sha256") == worker.get("model_sha256")
            ):
                raise ValueError(f"graph-context v2 member changed: {member}")
    elif not (
        training_exit == 2
        and probe_exit == 4
        and terminal.get("policy_audit_passed") is False
        and not members
        and not (root / "graph_context_development_probe.json").exists()
    ):
        raise ValueError("graph-context v2 failure is not a clean scientific rejection")

    return {
        "schema_version": 1,
        "status": "accepted" if training_exit == 0 else "scientifically_rejected",
        "run_id": "antelume-graph-context-frozen-ensemble-v2-verification",
        "training_exit_code": training_exit,
        "development_probe_exit_code": probe_exit,
        "deployment_member_count": len(members),
        "selection_ensemble": terminal.get("selection_ensemble"),
        "audit_ensemble": terminal.get("audit_ensemble"),
        "development_probe_metrics": None if probe is None else probe.get("metrics"),
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.archive) != args.archive_sha256:
        raise ValueError("graph-context v2 result archive hash changed")
    with tempfile.TemporaryDirectory(dir=args.output_root.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(args.archive, "r:gz") as archive:
            for member in archive.getmembers():
                safe_name(member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError("graph-context v2 archive contains an unsafe link/device")
            archive.extractall(staging, filter="data")
        result = verify_tree(staging / ROOT)
        if args.output_root.exists():
            raise ValueError("graph-context v2 extraction target already exists")
        shutil.move(str(staging / ROOT), args.output_root)
    args.terminal.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
