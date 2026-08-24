from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .ledger import (
    EventType,
    ExperimentEvent,
    Ledger,
    LedgerError,
    artifact_record,
    generate_run_id,
    git_state,
    registration_payload,
)
from .watch import collect_snapshot, persist_snapshot, status_line, write_status_reports


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_value(value: str | None) -> dict:
    if value is None:
        return {}
    candidate = Path(value)
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("configuration must be a JSON object")
    return parsed


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
    parser.error(f"unknown command: {args.command}")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except (LedgerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
