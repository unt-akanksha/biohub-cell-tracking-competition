from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download-competition-division-probe-frames.py"
SPEC = importlib.util.spec_from_file_location("competition_probe_frames", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def inventory() -> dict:
    movies = [
        {"stem": "movie_a", "required_frames": [2, 3, 4]},
        {"stem": "movie_b", "required_frames": [8, 9, 10]},
        {"stem": "movie_c", "required_frames": [12, 13, 14]},
        {"stem": "movie_d", "required_frames": [20, 21, 22, 30, 31, 32]},
    ]
    return {
        "schema_version": 1,
        "status": "competition_train_probe_only",
        "movies": movies,
        "summary": {
            "movies": 4,
            "event_frames": 5,
            "safe_recovery_positives": 3,
        },
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def test_required_files_are_train_only_and_collision_safe() -> None:
    movies = probe.validate_inventory(inventory())
    files = probe.required_files(movies)

    assert len(files) == 23
    assert len(files) == len(set(files))
    assert all(path.startswith("train/") for path in files)
    assert "train/movie_a.zarr/0/c/3/0/0/0" in files


def test_test_read_or_test_named_movie_is_rejected() -> None:
    payload = inventory()
    payload["competition_test_data_read"] = True
    with pytest.raises(ValueError, match="ineligible"):
        probe.validate_inventory(payload)

    payload = inventory()
    payload["movies"][0]["stem"] = "test_movie"
    with pytest.raises(ValueError, match="invalid train movie"):
        probe.validate_inventory(payload)


def test_download_files_records_exact_outputs(tmp_path: Path) -> None:
    files = ["train/movie_a.zarr/zarr.json", "train/movie_a.zarr/0/c/3/0/0/0"]
    commands: list[list[str]] = []

    def fake_runner(command: Sequence[str]) -> None:
        command = list(command)
        commands.append(command)
        remote = command[command.index("-f") + 1]
        destination = Path(command[command.index("-p") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / Path(remote).name).write_bytes(remote.encode("utf-8"))

    records = probe.download_files(
        files,
        tmp_path,
        competition="biohub",
        kaggle_executable="kaggle",
        force=True,
        runner=fake_runner,
    )

    assert [record["remote_path"] for record in records] == files
    assert all(record["bytes"] > 0 for record in records)
    assert all("--force" in command for command in commands)
