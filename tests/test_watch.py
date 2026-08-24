from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone

from biohub_tracker.cli import main
from biohub_tracker.kaggle import KaggleRunner
from biohub_tracker.watch import collect_snapshot, persist_snapshot, status_line


def test_tracer_snapshot_schema_and_status(tmp_path, fixture_dir, competition_config):
    snapshot = collect_snapshot(
        competition_config,
        fixture_dir=fixture_dir,
        now=datetime(2026, 8, 23, 20, 0, tzinfo=timezone.utc),
    )
    required = {
        "schema_version",
        "collected_at",
        "collection_status",
        "quota",
        "submissions",
        "leaderboard",
        "topics",
        "notebooks",
        "content_sha256",
    }
    assert required <= snapshot.keys()
    line = status_line(snapshot)
    assert "GPU remaining: 30.00h" in line
    assert "best public score: 0.913367" in line
    assert "notebooks reviewed: 2" in line

    first = persist_snapshot(snapshot, tmp_path)
    second = persist_snapshot(snapshot, tmp_path)
    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))["content_sha256"] == snapshot["content_sha256"]


def test_tracer_cli_fixture_mode(tmp_path, fixture_dir, capsys):
    (tmp_path / "config").mkdir()
    shutil.copy2("config/competition.json", tmp_path / "config" / "competition.json")
    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "watch",
            "--fixture-dir",
            str(fixture_dir.resolve()),
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "GPU remaining" in output
    assert "best public score" in output
    assert "notebooks reviewed" in output


def test_tracer_kaggle_runner_uses_argument_list_and_no_shell():
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "[]", "")

    KaggleRunner(run_process=fake_run).run_json(["quota", "--format", "json"])
    assert observed["command"] == ["kaggle", "quota", "--format", "json"]
    assert observed["kwargs"]["shell"] is False
