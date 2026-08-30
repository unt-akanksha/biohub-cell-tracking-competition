#!/usr/bin/env python
"""Download the minimal competition-train Zarr frames for the division probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence


RUN_ID = "competition-division-probe-frame-cache-v1"
DEFAULT_COMPETITION = "biohub-cell-tracking-during-development"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_inventory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = payload.get("summary", {})
    movies = payload.get("movies")
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "competition_train_probe_only"
        and payload.get("competition_train_data_read") is True
        and payload.get("competition_test_data_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and payload.get("authorized_for_submission") is False
        and isinstance(movies, list)
        and len(movies) == 4
        and summary.get("movies") == 4
        and summary.get("event_frames") == 5
        and summary.get("safe_recovery_positives") == 3
    ):
        raise ValueError("ineligible competition-train division probe inventory")
    for movie in movies:
        stem = movie.get("stem")
        frames = movie.get("required_frames")
        if (
            not isinstance(stem, str)
            or not stem
            or any(value in stem.lower() for value in ("test", "/", "\\"))
            or not isinstance(frames, list)
            or not frames
            or any(not isinstance(frame, int) or frame < 0 for frame in frames)
        ):
            raise ValueError("invalid train movie or frame inventory")
    return movies


def required_files(movies: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    for movie in movies:
        stem = movie["stem"]
        root = f"train/{stem}.zarr"
        files.extend((f"{root}/zarr.json", f"{root}/0/zarr.json"))
        files.extend(
            f"{root}/0/c/{int(frame)}/0/0/0"
            for frame in sorted(set(movie["required_frames"]))
        )
    if len(files) != len(set(files)):
        raise RuntimeError("competition frame file inventory contains duplicates")
    if any(not path.startswith("train/") for path in files):
        raise RuntimeError("competition probe attempted to read a non-train file")
    return files


def default_runner(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def download_files(
    files: list[str],
    output_root: Path,
    *,
    competition: str,
    kaggle_executable: str,
    force: bool,
    runner: Callable[[Sequence[str]], None] = default_runner,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for remote_path in files:
        target = output_root / Path(remote_path)
        if force or not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [
                kaggle_executable,
                "competitions",
                "download",
                "-c",
                competition,
                "-f",
                remote_path,
                "-p",
                str(target.parent),
            ]
            if force:
                command.append("--force")
            runner(command)
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"Kaggle did not produce {target}")
        records.append(
            {
                "remote_path": remote_path,
                "local_relative_path": target.relative_to(output_root).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return records


def validate_zarr_metadata(output_root: Path, movies: list[dict[str, Any]]) -> None:
    for movie in movies:
        root = output_root / "train" / f"{movie['stem']}.zarr"
        group = json.loads((root / "zarr.json").read_text(encoding="utf-8"))
        array = json.loads((root / "0" / "zarr.json").read_text(encoding="utf-8"))
        if group.get("zarr_format") != 3 or group.get("node_type") != "group":
            raise RuntimeError(f"unexpected Zarr group metadata for {movie['stem']}")
        shape = array.get("shape")
        if (
            array.get("zarr_format") != 3
            or array.get("node_type") != "array"
            or not isinstance(shape, list)
            or len(shape) != 4
            or any(frame >= int(shape[0]) for frame in movie["required_frames"])
        ):
            raise RuntimeError(f"unexpected Zarr array metadata for {movie['stem']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--kaggle-executable", default="kaggle")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    movies = validate_inventory(inventory)
    files = required_files(movies)
    records = download_files(
        files,
        args.output_root,
        competition=args.competition,
        kaggle_executable=args.kaggle_executable,
        force=args.force,
    )
    validate_zarr_metadata(args.output_root, movies)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "competition": args.competition,
        "files": records,
        "summary": {
            "files": len(records),
            "bytes": sum(record["bytes"] for record in records),
            "movies": len(movies),
            "frames": sum(len(movie["required_frames"]) for movie in movies),
        },
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "probe_cache_manifest.json", manifest)
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
