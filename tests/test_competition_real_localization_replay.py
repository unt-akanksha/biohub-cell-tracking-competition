from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
from typing import Sequence

import pytest

from research.build_competition_real_localization_inventory import (
    fallback_center,
    selected_nondivision_stems,
    split_selection_roles,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download-competition-real-localization-replay.py"
SPEC = importlib.util.spec_from_file_location("real_localization_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def fixture_movies() -> list[dict]:
    rows: list[dict] = []
    for embryo in ("44b6", "6bba"):
        for index in range(18):
            rows.append(
                {
                    "stem": f"{embryo}_opt_{index:02d}",
                    "embryo": embryo,
                    "role": "optimization",
                    "event_timepoints": [10] if index < 2 else [],
                }
            )
        for index in range(4):
            rows.append(
                {
                    "stem": f"{embryo}_sel_{index:02d}",
                    "embryo": embryo,
                    "role": "selection",
                    "event_timepoints": [20],
                }
            )
    return rows


def test_role_splits_are_deterministic_balanced_and_nonoverlapping() -> None:
    movies = fixture_movies()
    first = split_selection_roles(movies)
    second = split_selection_roles(list(reversed(movies)))
    assert first == second
    assert set(first.values()) == {"selection", "sealed_audit"}
    for embryo in ("44b6", "6bba"):
        values = [role for stem, role in first.items() if stem.startswith(embryo)]
        assert values.count("selection") == 2
        assert values.count("sealed_audit") == 2
    nondivision = selected_nondivision_stems(movies)
    assert len(nondivision) == 32
    assert all("opt" in stem for stem in nondivision)


def test_fallback_center_is_interior_stable_and_annotated() -> None:
    nodes = {
        1: (0.0, 1.0, 1.0, 1.0),
        2: (4.0, 1.0, 1.0, 1.0),
        3: (8.0, 1.0, 1.0, 1.0),
        4: (99.0, 1.0, 1.0, 1.0),
    }
    center = fallback_center("44b6_example", nodes)
    assert center in {4, 8}
    assert fallback_center("44b6_example", nodes) == center


def test_fallback_center_excludes_graph_boundaries() -> None:
    nodes = {
        1: (0.0, 1.0, 1.0, 1.0),
        2: (99.0, 1.0, 1.0, 1.0),
    }
    with pytest.raises(ValueError, match="no interior"):
        fallback_center("44b6_boundary_only", nodes)


def replay_inventory() -> dict:
    movies = [
        {
            "stem": "44b6_opt",
            "role": "optimization",
            "center_frames": [3],
            "required_frames": [2, 3, 4],
        },
        {
            "stem": "44b6_select",
            "role": "selection",
            "center_frames": [8],
            "required_frames": [7, 8, 9],
        },
        {
            "stem": "6bba_audit",
            "role": "sealed_audit",
            "center_frames": [12],
            "required_frames": [11, 12, 13],
        },
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": "competition-real-localization-inventory-v1",
        "movies": movies,
        "summary": {"movies": 3},
        "excluded_final_probe_stems": sorted(replay.FINAL_PROBE_STEMS),
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def test_downloader_inventory_and_files_are_train_only() -> None:
    movies = replay.validate_inventory(replay_inventory())
    files = replay.required_files(movies)
    assert len(files) == 15
    assert len(files) == len(set(files))
    assert all(path.startswith("train/") for path in files)
    assert files[0].startswith("train/44b6_select.zarr/")
    assert "train/44b6_select.zarr/0/c/8/0/0/0" in files


def test_downloader_rejects_probe_or_test_data() -> None:
    payload = replay_inventory()
    payload["competition_test_data_read"] = True
    with pytest.raises(ValueError, match="ineligible"):
        replay.validate_inventory(payload)
    payload = replay_inventory()
    payload["movies"][0]["stem"] = "test_movie"
    with pytest.raises(ValueError, match="invalid train-only"):
        replay.validate_inventory(payload)


def test_download_records_exact_hashes(tmp_path: Path) -> None:
    files = ["train/44b6_opt.zarr/zarr.json", "train/44b6_opt.zarr/0/c/3/0/0/0"]
    commands: list[list[str]] = []

    def fake_runner(command: Sequence[str]) -> None:
        command = list(command)
        commands.append(command)
        remote = command[command.index("-f") + 1]
        destination = Path(command[command.index("-p") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / Path(remote).name).write_bytes(remote.encode("utf-8"))

    records = replay.download_files(
        files,
        tmp_path,
        competition="biohub",
        kaggle_executable="kaggle",
        force=True,
        runner=fake_runner,
    )
    assert [record["remote_path"] for record in records] == files
    assert all(len(record["sha256"]) == 64 for record in records)
    assert all("--quiet" in command and "--force" in command for command in commands)


def test_default_runner_retries_429_with_bounded_backoff(monkeypatch) -> None:
    responses = [
        subprocess.CompletedProcess(
            ["kaggle"], 1, stdout="", stderr="429 Too Many Requests"
        ),
        subprocess.CompletedProcess(["kaggle"], 0, stdout="", stderr=""),
    ]
    sleeps: list[int] = []

    def fake_run(*_args, **_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(replay.subprocess, "run", fake_run)
    monkeypatch.setattr(replay.time, "sleep", sleeps.append)
    replay.default_runner(["kaggle", "competitions", "download"])
    assert sleeps == [120]


def test_default_runner_does_not_retry_permanent_failure(monkeypatch) -> None:
    response = subprocess.CompletedProcess(
        ["kaggle"], 1, stdout="", stderr="403 Permission denied"
    )
    monkeypatch.setattr(replay.subprocess, "run", lambda *_args, **_kwargs: response)
    with pytest.raises(subprocess.CalledProcessError):
        replay.default_runner(["kaggle", "competitions", "download"])
