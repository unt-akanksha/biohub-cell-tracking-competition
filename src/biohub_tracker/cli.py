from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from .guard import GuardInputError, evaluate_guard, list_active_gpu_kernels, read_gpu_quota
from .kaggle import FixtureRunner, KaggleRunner

from .ledger import (
    EventType,
    ExperimentEvent,
    Ledger,
    LedgerError,
    amendment_payload,
    artifact_record,
    completed_payload,
    decision_payload,
    failed_payload,
    generate_run_id,
    git_state,
    rejected_payload,
    registration_payload,
    reconstruct_runs,
    start_payload,
)
from .progress import render_progress_json, render_progress_markdown, write_progress_reports
from .watch import collect_snapshot, persist_snapshot, status_line, write_status_reports


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_value(value: str | None) -> dict:
    if value is None:
        return {}
    candidate = Path(value)
    if not value.lstrip().startswith("{") and candidate.is_file():
        with candidate.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("configuration must be a JSON object")
    return parsed


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_float=Decimal)
    if not isinstance(value, dict):
        raise ValueError("metrics report must be a JSON object")
    return value


def _artifact_records(root: Path, paths: Sequence[Path] | None) -> list[dict[str, str]]:
    return [artifact_record(root, path) for path in (paths or [])]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biohub", description="Biohub competition control plane")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    watch = subparsers.add_parser("watch", help="capture read-only competition state")
    source = watch.add_mutually_exclusive_group()
    source.add_argument("--live", action="store_true", help="read the authenticated Kaggle CLI")
    source.add_argument("--fixture-dir", type=Path, help="read deterministic JSON fixtures")
    watch.add_argument("--top", type=int, default=20, help="number of public notebooks to inspect")
    watch.add_argument(
        "--audit-notebook-sources",
        action="store_true",
        help="pull and statically inspect notebook source without executing it",
    )

    experiment = subparsers.add_parser("experiment", help="append immutable experiment events")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    register = experiment_commands.add_parser("register", help="register an experiment")
    register.add_argument("--run-id")
    register.add_argument("--hypothesis", required=True)
    register.add_argument("--max-runtime-hours", required=True)
    register.add_argument("--parent")
    register.add_argument("--config")
    register.add_argument("--seed", action="append", type=int, default=[])
    register.add_argument("--split", default="not-registered")
    register.add_argument("--data-path", type=Path)
    register.add_argument("--data-sha256")
    register.add_argument("--model-path", type=Path)
    register.add_argument("--model-sha256")
    repair = experiment_commands.add_parser("repair", help="acknowledge a quarantined tail")
    repair.add_argument("--reason", required=True)
    start = experiment_commands.add_parser("start", help="record a launched experiment")
    start.add_argument("run_id")
    start.add_argument("--kaggle-ref", required=True)
    start.add_argument("--authorization-id", required=True)
    start.add_argument("--quota-before-hours", required=True)
    finish = experiment_commands.add_parser("finish", help="record a completed experiment")
    finish.add_argument("run_id")
    finish.add_argument("--actual-runtime-hours", required=True)
    finish.add_argument("--quota-after-hours", required=True)
    finish.add_argument("--metrics-report", type=Path, required=True)
    finish.add_argument("--artifact", type=Path, action="append", default=[])
    finish.add_argument("--report", type=Path, action="append", default=[])
    finish.add_argument("--public-score")
    fail = experiment_commands.add_parser("fail", help="record an experiment failure")
    fail.add_argument("run_id")
    fail.add_argument("--actual-runtime-hours", required=True)
    fail.add_argument("--quota-after-hours", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--traceback", type=Path)
    reject = experiment_commands.add_parser("reject", help="record a failed experiment gate")
    reject.add_argument("run_id")
    reject.add_argument("--actual-runtime-hours", required=True)
    reject.add_argument("--quota-after-hours", required=True)
    reject.add_argument("--gate", required=True)
    reject.add_argument("--reason", required=True)
    decide = experiment_commands.add_parser("decide", help="record a promotion decision")
    decide.add_argument("run_id")
    decide.add_argument(
        "--decision",
        choices=["promote", "retain", "retire", "inconclusive"],
        required=True,
    )
    decide.add_argument("--evidence", action="append", required=True)
    amend = experiment_commands.add_parser("amend", help="append a correction without mutation")
    amend.add_argument("run_id")
    amend.add_argument("--target-event-id", required=True)
    amend.add_argument("--reason", required=True)
    amend.add_argument("--replacement", required=True)
    progress = subparsers.add_parser("progress", help="render immutable experiment lineage")
    progress.add_argument("--json", action="store_true", dest="json_output")
    guard = subparsers.add_parser("guard", help="evaluate quota-safe Kaggle launch eligibility")
    guard.add_argument("--run-id", required=True)
    guard.add_argument("--max-runtime-hours", required=True)
    guard_source = guard.add_mutually_exclusive_group()
    guard_source.add_argument("--live", action="store_true", help="read authenticated Kaggle state")
    guard_source.add_argument("--fixture-dir", type=Path, help="read deterministic fixtures")
    guard.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "watch":
        fixture_dir = args.fixture_dir
        if not args.live and fixture_dir is None:
            fixture_dir = root / "tests" / "fixtures" / "kaggle"
        config = _load_config(root / "config" / "competition.json")
        snapshot = collect_snapshot(
            config,
            fixture_dir=fixture_dir,
            live=args.live,
            notebook_limit=args.top,
            root=root,
            audit_notebook_sources=args.audit_notebook_sources,
        )
        path = persist_snapshot(snapshot, root)
        report_path, _ = write_status_reports(snapshot, config, root)
        print(path)
        print(report_path)
        print(status_line(snapshot))
        return 0
    if args.command == "experiment":
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        if args.experiment_command == "register":
            run_id = args.run_id or generate_run_id(args.hypothesis)
            data = (
                artifact_record(root, args.data_path, expected_sha256=args.data_sha256)
                if args.data_path
                else ({"sha256": args.data_sha256} if args.data_sha256 else None)
            )
            model = (
                artifact_record(root, args.model_path, expected_sha256=args.model_sha256)
                if args.model_path
                else ({"sha256": args.model_sha256} if args.model_sha256 else None)
            )
            payload = registration_payload(
                hypothesis=args.hypothesis,
                parent=args.parent,
                config=_json_value(args.config),
                seeds=args.seed,
                split=args.split,
                declared_max_runtime_hours=args.max_runtime_hours,
                code=git_state(root),
                data_artifact=data,
                model_artifact=model,
            )
            ledger.append(ExperimentEvent.create(run_id, EventType.REGISTERED, payload))
            print(run_id)
            return 0
        if args.experiment_command == "repair":
            print(ledger.repair_truncated(args.reason))
            return 0
        if args.experiment_command == "start":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.STARTED,
                start_payload(
                    kaggle_ref=args.kaggle_ref,
                    authorization_id=args.authorization_id,
                    quota_before_hours=args.quota_before_hours,
                ),
            )
        elif args.experiment_command == "finish":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.COMPLETED,
                completed_payload(
                    actual_runtime_hours=args.actual_runtime_hours,
                    quota_after_hours=args.quota_after_hours,
                    metrics=_load_metrics(args.metrics_report),
                    artifacts=_artifact_records(root, args.artifact),
                    reports=_artifact_records(root, args.report),
                    public_score=args.public_score,
                ),
            )
        elif args.experiment_command == "fail":
            traceback_artifact = artifact_record(root, args.traceback) if args.traceback else None
            event = ExperimentEvent.create(
                args.run_id,
                EventType.FAILED,
                failed_payload(
                    actual_runtime_hours=args.actual_runtime_hours,
                    quota_after_hours=args.quota_after_hours,
                    failure_reason=args.reason,
                    traceback_artifact=traceback_artifact,
                ),
            )
        elif args.experiment_command == "reject":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.REJECTED,
                rejected_payload(
                    actual_runtime_hours=args.actual_runtime_hours,
                    quota_after_hours=args.quota_after_hours,
                    failed_gate=args.gate,
                    reason=args.reason,
                ),
            )
        elif args.experiment_command == "decide":
            event = ExperimentEvent.create(
                args.run_id,
                EventType.DECISION,
                decision_payload(args.decision, args.evidence),
            )
        elif args.experiment_command == "amend":
            replacement = _json_value(args.replacement)
            target = next(
                (
                    candidate
                    for candidate in ledger.read_events()
                    if candidate.event_id == args.target_event_id and candidate.run_id == args.run_id
                ),
                None,
            )
            if target is None:
                raise ValueError("amendment target was not found for this run")
            event = ExperimentEvent.create(
                args.run_id,
                EventType.AMENDMENT,
                amendment_payload(
                    target_event_id=args.target_event_id,
                    correction_reason=args.reason,
                    replacement_fields=replacement,
                ),
            )
            ledger.append(event)
            print(
                json.dumps(
                    {"original": target.payload, "corrected": replacement, "event_id": event.event_id},
                    sort_keys=True,
                )
            )
            return 0
        else:
            parser.error(f"unknown experiment command: {args.experiment_command}")
        ledger.append(event)
        print(event.event_id)
        return 0
    if args.command == "progress":
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        events = ledger.read_events()
        markdown_path, _ = write_progress_reports(events, root)
        if args.json_output:
            print(json.dumps(render_progress_json(events), indent=2, sort_keys=True))
        else:
            print(render_progress_markdown(events))
            print(f"\nReport: {markdown_path}")
        return 0
    if args.command == "guard":
        config = _load_config(root / "config" / "competition.json")
        ledger = Ledger(root / "experiments" / "events.jsonl", root)
        runs = reconstruct_runs(ledger.read_events())
        state = runs.get(args.run_id)
        fixture_dir = args.fixture_dir or (root / "tests" / "fixtures" / "kaggle")
        runner = KaggleRunner() if args.live else FixtureRunner(fixture_dir)
        quota = None
        active = []
        input_error = None
        try:
            quota = read_gpu_quota(runner)
            active = list_active_gpu_kernels(runner, config["slug"])
        except GuardInputError as exc:
            input_error = exc.reason_code
        decision = evaluate_guard(
            run_id=args.run_id,
            registered_status=state.status if state else None,
            declared_max_runtime=args.max_runtime_hours,
            quota=quota,
            active_kernels=active,
            reserve=config["gpu_reserve_hours"],
            notebook_maximum=config["notebook_runtime_limit_hours"],
            input_error_code=input_error,
        )
        if state:
            ledger.append(
                ExperimentEvent.create(args.run_id, EventType.GUARD_DECISION, decision.to_dict())
            )
        print(json.dumps(decision.to_dict(), indent=2 if args.json_output else None, sort_keys=True))
        return 0 if decision.authorized else 2
    parser.error(f"unknown command: {args.command}")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except (GuardInputError, LedgerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
