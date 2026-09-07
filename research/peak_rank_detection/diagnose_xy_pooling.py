#!/usr/bin/env python
"""Compare XY reduction methods on optimization-only raw Biohub frames.

The expanded replay uses phase-locked ``::4`` decimation. This diagnostic
joins only optimization members to already downloaded raw train chunks,
verifies the current decimation byte-for-byte, and compares signal-preserving
4x4 reductions without reading selection, sealed-audit, test, or leaderboard
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import blosc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.peak_rank_detection.diagnose_multiscale_blob_prior import (
    difference_of_averages,
    normalize_volume,
    peak_coordinates,
    point_distances,
    summarize,
)

MEMBER_PATTERN = re.compile(
    r"(?P<stem>(?:44b6|6bba)_[0-9a-f]+)__t(?P<time>\d{4})\.npz$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xy_reductions(frame: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(frame, dtype=np.float32)
    if values.shape != (64, 256, 256):
        raise ValueError("raw frame must have shape (64,256,256)")
    blocks = values.reshape(64, 64, 4, 64, 4)
    mean = blocks.mean(axis=(2, 4))
    maximum = blocks.max(axis=(2, 4))
    rms = np.sqrt(np.square(blocks).mean(axis=(2, 4)))
    return {
        "phase00_decimate": values[:, ::4, ::4],
        "area_mean4": mean,
        "area_rms4": rms,
        "area_mean75_max25": 0.75 * mean + 0.25 * maximum,
        "area_mean50_max50": 0.50 * mean + 0.50 * maximum,
        "area_max4": maximum,
    }


def raw_frame(path: Path) -> np.ndarray:
    decoded = blosc.decompress(path.read_bytes())
    expected = 64 * 256 * 256
    values = np.frombuffer(decoded, dtype="<u2")
    if values.size != expected:
        raise ValueError(f"unexpected raw chunk size: {path}")
    return values.reshape(64, 256, 256)


def truth_points(nodes: np.ndarray) -> np.ndarray:
    rows = np.asarray(nodes, dtype=np.float32)
    points = rows[rows[:, 0].astype(np.int64) == 1, 1:4].copy()
    points[:, 1:] /= 4.0
    if not len(points) or np.any(points < 0.0) or np.any(points >= 64.0):
        raise ValueError("optimization truth is empty or out of bounds")
    return points


def available_chunk(raw_root: Path, stem: str, timepoint: int) -> Path | None:
    path = raw_root / f"{stem}.zarr" / "0" / "c" / str(timepoint) / "0" / "0" / "0"
    return path if path.is_file() else None


def evaluate(
    archive: Path,
    raw_root: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("expanded replay archive hash changed")
    distances: dict[str, list[float]] = {}
    embryo_distances: dict[str, dict[str, list[float]]] = {}
    embryo_crops: dict[str, int] = {}
    crops = 0
    with tarfile.open(archive, "r") as bundle:
        members = sorted(
            (
                member
                for member in bundle.getmembers()
                if member.isfile()
                and "/optimization/" in member.name
                and member.name.endswith(".npz")
            ),
            key=lambda member: member.name,
        )
        if len(members) != 480:
            raise ValueError("expanded replay must contain 480 optimization crops")
        for member in members:
            if "/selection/" in member.name or "/sealed_audit/" in member.name:
                raise ValueError("diagnostic crossed the optimization boundary")
            match = MEMBER_PATTERN.search(member.name)
            if match is None:
                raise ValueError(f"unexpected optimization member: {member.name}")
            chunk = available_chunk(raw_root, match["stem"], int(match["time"]))
            if chunk is None:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable optimization member: {member.name}")
            with np.load(io.BytesIO(stream.read()), allow_pickle=False) as payload:
                current = np.asarray(payload["volumes"])[1]
                truth = truth_points(payload["nodes"])
            reductions = xy_reductions(raw_frame(chunk))
            if not np.array_equal(reductions["phase00_decimate"], current):
                raise ValueError(
                    "raw chunk does not reproduce the frozen decimated crop"
                )
            embryo = match["stem"].split("_", 1)[0]
            embryo_crops[embryo] = embryo_crops.get(embryo, 0) + 1
            for name, values in reductions.items():
                normalized = normalize_volume(values)
                response = difference_of_averages(normalized, 5, 13)
                predicted = peak_coordinates(response, maximum_predictions=64)
                measured = point_distances(predicted, truth)
                distances.setdefault(name, []).extend(map(float, measured))
                embryo_distances.setdefault(name, {}).setdefault(embryo, []).extend(
                    map(float, measured)
                )
            crops += 1
    if crops < 100:
        raise ValueError("fewer than 100 optimization crops joined to raw chunks")
    rows = []
    for name, values in distances.items():
        row = {"reduction": name, **summarize(values, crops=crops)}
        row["embryos"] = {
            embryo: summarize(items, crops=embryo_crops[embryo])
            for embryo, items in sorted(embryo_distances[name].items())
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["top64_recall_at_2_5_voxels"],
            row["mean_capped_distance_voxels"],
            row["reduction"],
        )
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "optimization_only_raw_xy_reduction",
        "archive_sha256": actual_sha256,
        "joined_optimization_crops": crops,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "selection_data_read": False,
        "sealed_audit_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "response": "current_frame_avg5_minus_avg13",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.archive.resolve(),
        args.raw_root.resolve(),
        expected_sha256=args.expected_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
