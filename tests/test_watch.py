from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracker.cli import main
from biohub_tracker.kaggle import KaggleRunner
from biohub_tracker.watch import (
    build_status_projection,
    collect_snapshot,
    persist_snapshot,
    render_status_report,
    status_line,
)


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


def test_full_leaderboard_download_is_parsed_without_extracting(tmp_path):
    def fake_run(command, **kwargs):
        directory = Path(command[command.index("--path") + 1])
        with zipfile.ZipFile(directory / "leaderboard.zip", "w") as archive:
            archive.writestr(
                "leaderboard.csv",
                "TeamId,TeamName,SubmissionDate,Score\n747,IndarKumar,2026-08-23,0.913367\n",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    rows = KaggleRunner(run_process=fake_run).download_leaderboard(
        "biohub-cell-tracking-during-development"
    )
    assert rows[0]["TeamName"] == "IndarKumar"
    assert rows[0]["Score"] == "0.913367"


def test_report_contains_required_status_and_quota_math(fixture_dir, competition_config):
    snapshot = collect_snapshot(
        competition_config,
        fixture_dir=fixture_dir,
        root=Path.cwd(),
        now=datetime(2026, 8, 23, 20, 0, tzinfo=timezone.utc),
    )
    report = render_status_report(snapshot, competition_config)
    for section in (
        "## Competition",
        "## GPU Safety",
        "## Personal Submissions",
        "## Notebook Provenance",
        "## Recent Discussions",
        "## Next Gate",
    ):
        assert section in report
    assert "Spendable before reserve: 22.00 hours" in report
    assert "`xiaoleilian/biohub-ct-mix-divaug`" in report
    assert "Excluded Metric Hacks" in report


def test_report_is_deterministic_and_missing_rank_is_explicit(fixture_dir, competition_config):
    snapshot = collect_snapshot(
        competition_config,
        fixture_dir=fixture_dir,
        root=Path.cwd(),
        now=datetime(2026, 8, 23, 20, 0, tzinfo=timezone.utc),
    )
    snapshot["leaderboard"] = [
        row for row in snapshot["leaderboard"] if row["team"] != "IndarKumar"
    ]
    first = render_status_report(snapshot, competition_config)
    second = render_status_report(snapshot, competition_config)
    assert first == second
    assert "Public rank: unavailable (configured team alias is absent" in first
    projection = build_status_projection(snapshot, competition_config)
    assert projection["competition"]["rank"] is None
    assert projection["gpu"]["spendable_hours"] == "22.00"
