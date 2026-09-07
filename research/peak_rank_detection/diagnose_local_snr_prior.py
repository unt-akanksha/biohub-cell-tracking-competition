#!/usr/bin/env python
"""Screen locally normalized blob evidence on real optimization crops only."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter

from research.peak_rank_detection.diagnose_multiscale_blob_prior import (
    difference_of_averages,
    embryo_from_member_name,
    center_points,
    normalize_volume,
    peak_coordinates,
    point_distances,
    sha256_file,
    standardized_response,
    summarize,
)


def local_snr_band(values: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Return a center-surround response normalized by fixed local variation."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("values must be a 3D volume")
    surround_mean = uniform_filter(values, size=outer, mode="nearest")
    surround_square_mean = uniform_filter(
        np.square(values), size=outer, mode="nearest"
    )
    variance = np.maximum(surround_square_mean - np.square(surround_mean), 0.0)
    local_scale = np.sqrt(variance).astype(np.float32, copy=False)
    positive_scale = local_scale[local_scale > 0.0]
    floor = (
        max(float(np.quantile(positive_scale, 0.25)), 1e-3)
        if len(positive_scale)
        else 1e-3
    )
    return difference_of_averages(values, inner, outer) / np.maximum(
        local_scale, floor
    )


def candidate_response_maps(volumes: np.ndarray) -> dict[str, np.ndarray]:
    """Build fixed scale and temporal-reducer candidates without labels."""

    volumes = np.asarray(volumes)
    if volumes.shape != (3, 64, 64, 64):
        raise ValueError("volumes must have shape (3,64,64,64)")
    normalized = normalize_volume(volumes)
    current = normalized[1]
    reducers = {
        "mean": normalized.mean(axis=0),
        "median": np.median(normalized, axis=0),
        "minimum": normalized.min(axis=0),
        "maximum": normalized.max(axis=0),
    }
    maps: dict[str, np.ndarray] = {}
    for reducer_name, reduced in reducers.items():
        fixed = []
        snr = []
        for inner, outer in ((3, 9), (3, 11), (5, 13)):
            fixed.extend(
                (
                    standardized_response(
                        difference_of_averages(current, inner, outer)
                    ),
                    standardized_response(
                        difference_of_averages(reduced, inner, outer)
                    ),
                )
            )
            snr.extend(
                (
                    standardized_response(local_snr_band(current, inner, outer)),
                    standardized_response(local_snr_band(reduced, inner, outer)),
                )
            )
        maps[f"{reducer_name}:fixed-multiscale"] = np.mean(fixed, axis=0)
        maps[f"{reducer_name}:local-snr-multiscale"] = np.mean(snr, axis=0)
    return maps


def evaluate_archive(archive: Path, *, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("expanded replay archive hash changed")
    accumulated: dict[str, list[float]] = {}
    by_embryo: dict[str, dict[str, list[float]]] = {}
    embryo_crops = {"44b6": 0, "6bba": 0}
    with tarfile.open(archive, mode="r") as bundle:
        members = sorted(
            (
                member
                for member in bundle.getmembers()
                if member.isfile() and member.name.endswith(".npz")
            ),
            key=lambda member: member.name,
        )
        optimization = [member for member in members if "/optimization/" in member.name]
        if len(optimization) != 480:
            raise ValueError("expanded replay must contain 480 optimization crops")
        for member in optimization:
            if "/selection/" in member.name or "/sealed_audit/" in member.name:
                raise ValueError("diagnostic attempted to cross the optimization boundary")
            embryo = embryo_from_member_name(member.name)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member: {member.name}")
            with np.load(io.BytesIO(extracted.read()), allow_pickle=False) as payload:
                truth = center_points(payload["nodes"])
                maps = candidate_response_maps(payload["volumes"])
            for name, response in maps.items():
                predicted = peak_coordinates(response, maximum_predictions=64)
                distances = point_distances(predicted, truth)
                accumulated.setdefault(name, []).extend(float(x) for x in distances)
                by_embryo.setdefault(name, {}).setdefault(embryo, []).extend(
                    float(x) for x in distances
                )
            embryo_crops[embryo] += 1
    rows = []
    for name, distances in accumulated.items():
        row = {"response": name, **summarize(distances, crops=480)}
        row["embryos"] = {
            embryo: summarize(by_embryo[name][embryo], crops=embryo_crops[embryo])
            for embryo in sorted(embryo_crops)
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -min(
                row["embryos"][embryo]["top64_recall_at_2_5_voxels"]
                for embryo in embryo_crops
            ),
            -row["top64_recall_at_2_5_voxels"],
            row["response"],
        )
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "optimization_only_local_snr_and_temporal_blob_prior",
        "archive_sha256": actual_sha256,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "selection_data_read": False,
        "sealed_audit_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "maximum_predictions_per_crop": 64,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_archive(
        args.archive.resolve(), expected_sha256=args.expected_sha256
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report["rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
