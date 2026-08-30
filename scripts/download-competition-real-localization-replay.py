#!/usr/bin/env python
"""Download a frozen train-only Biohub localization replay inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Sequence


RUN_ID = "competition-real-localization-frame-cache-v1"
INVENTORY_RUN_ID = "competition-real-localization-inventory-v1"
DEFAULT_COMPETITION = "biohub-cell-tracking-during-development"
FINAL_PROBE_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}
MAXIMUM_DOWNLOAD_ATTEMPTS = 8
BASE_RETRY_SECONDS = 5
RATE_LIMIT_RETRY_SECONDS = 120


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
    movies = payload.get("movies")
    summary = payload.get("summary", {})
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "complete"
        and payload.get("run_id") == INVENTORY_RUN_ID
        and payload.get("excluded_final_probe_stems") == sorted(FINAL_PROBE_STEMS)
        and payload.get("competition_train_data_read") is True
        and payload.get("competition_test_data_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and payload.get("authorized_for_submission") is False
        and isinstance(movies, list)
        and movies
        and summary.get("movies") == len(movies)
    ):
        raise ValueError("ineligible real localization inventory")
    seen: set[str] = set()
    for movie in movies:
        stem = movie.get("stem")
        centers = movie.get("center_frames")
        frames = movie.get("required_frames")
        role = movie.get("role")
        expected_frames = sorted(
            {
                int(center) + offset
                for center in centers or []
                for offset in (-1, 0, 1)
            }
        )
        if (
            not isinstance(stem, str)
            or not stem
            or stem in FINAL_PROBE_STEMS
            or any(value in stem.lower() for value in ("test", "/", "\\"))
            or stem in seen
            or role not in {"optimization", "selection", "sealed_audit"}
            or not isinstance(centers, list)
            or not centers
            or not isinstance(frames, list)
            or frames != expected_frames
            or any(not isinstance(frame, int) or frame < 0 for frame in frames)
        ):
            raise ValueError("invalid train-only real localization movie")
        seen.add(stem)
    if {movie["role"] for movie in movies} != {
        "optimization",
        "selection",
        "sealed_audit",
    }:
        raise ValueError("real localization inventory lost a role")
    return movies


def required_files(movies: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    # Gate shards are small in count and scientifically blocking, so resume
    # them before the larger optimization pool after any API throttle window.
    role_priority = {"selection": 0, "sealed_audit": 1, "optimization": 2}
    for movie in sorted(movies, key=lambda row: (role_priority[row["role"]], row["stem"])):
        root = f"train/{movie['stem']}.zarr"
        files.extend((f"{root}/zarr.json", f"{root}/0/zarr.json"))
        files.extend(
            f"{root}/0/c/{int(frame)}/0/0/0"
            for frame in movie["required_frames"]
        )
    if len(files) != len(set(files)) or any(
        not path.startswith("train/") or "/test/" in path.lower() for path in files
    ):
        raise RuntimeError("real localization file inventory is not train-only")
    return files


def default_runner(command: Sequence[str]) -> None:
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, MAXIMUM_DOWNLOAD_ATTEMPTS + 1):
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
        last = completed
        if completed.returncode == 0:
            return
        output = f"{completed.stdout}\n{completed.stderr}".lower()
        transient = any(
            token in output
            for token in (
                "429",
                "too many requests",
                "500 internal server error",
                "502 bad gateway",
                "503 service unavailable",
                "504 gateway timeout",
                "connection reset",
                "timed out",
            )
        )
        if not transient or attempt == MAXIMUM_DOWNLOAD_ATTEMPTS:
            raise subprocess.CalledProcessError(
                completed.returncode,
                list(command),
                output=completed.stdout,
                stderr=completed.stderr,
            )
        rate_limited = "429" in output or "too many requests" in output
        base_delay = RATE_LIMIT_RETRY_SECONDS if rate_limited else BASE_RETRY_SECONDS
        delay = min(300, base_delay * 2 ** (attempt - 1))
        print(
            f"transient Kaggle download failure; retry {attempt + 1}/"
            f"{MAXIMUM_DOWNLOAD_ATTEMPTS} in {delay}s",
            flush=True,
        )
        time.sleep(delay)
    if last is None:  # pragma: no cover - the bounded loop always executes
        raise RuntimeError("download retry loop did not execute")


def download_files(
    files: list[str],
    output_root: Path,
    *,
    competition: str,
    kaggle_executable: str,
    force: bool,
    request_delay_seconds: float = 0.0,
    runner: Callable[[Sequence[str]], None] = default_runner,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, remote_path in enumerate(files, start=1):
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
                "--quiet",
            ]
            if force:
                command.append("--force")
            runner(command)
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
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
        if index % 50 == 0:
            print(f"downloaded {index}/{len(files)} train-only files", flush=True)
    return records


def validate_zarr_metadata(output_root: Path, movies: list[dict[str, Any]]) -> None:
    for movie in movies:
        root = output_root / "train" / f"{movie['stem']}.zarr"
        group = json.loads((root / "zarr.json").read_text(encoding="utf-8"))
        array = json.loads((root / "0" / "zarr.json").read_text(encoding="utf-8"))
        shape = array.get("shape")
        if (
            group.get("zarr_format") != 3
            or group.get("node_type") != "group"
            or array.get("zarr_format") != 3
            or array.get("node_type") != "array"
            or not isinstance(shape, list)
            or len(shape) != 4
            or tuple(shape[1:]) != (64, 256, 256)
            or any(frame >= int(shape[0]) for frame in movie["required_frames"])
        ):
            raise RuntimeError(f"unexpected train Zarr metadata for {movie['stem']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--kaggle-executable", default="kaggle")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--request-delay-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 <= args.request_delay_seconds <= 10.0:
        raise ValueError("request delay must be between zero and ten seconds")

    if sha256_file(args.inventory) != args.expected_inventory_sha256.lower():
        raise ValueError("real localization inventory hash changed")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    movies = validate_inventory(inventory)
    files = required_files(movies)
    records = download_files(
        files,
        args.output_root,
        competition=args.competition,
        kaggle_executable=args.kaggle_executable,
        force=args.force,
        request_delay_seconds=args.request_delay_seconds,
    )
    validate_zarr_metadata(args.output_root, movies)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "competition": args.competition,
        "inventory_sha256": sha256_file(args.inventory),
        "movies": movies,
        "files": records,
        "summary": {
            "files": len(records),
            "bytes": sum(record["bytes"] for record in records),
            "movies": len(movies),
            "center_frames": sum(len(movie["center_frames"]) for movie in movies),
            "frames": sum(len(movie["required_frames"]) for movie in movies),
            "by_role": inventory["summary"]["by_role"],
        },
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "real_localization_frame_cache_manifest.json", manifest)
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
