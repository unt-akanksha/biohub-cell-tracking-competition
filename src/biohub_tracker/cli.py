from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .watch import collect_snapshot, persist_snapshot, status_line


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
        print(path)
        print(status_line(snapshot))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2
