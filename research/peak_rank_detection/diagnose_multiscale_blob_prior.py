#!/usr/bin/env python
"""Measure scale-space blob priors on the clean real optimization role only.

This diagnostic is deliberately model-free and training-role-only.  It reads
the hash-pinned expanded replay archive without extracting it, refuses every
member outside ``optimization/``, and reports top-k localization coverage for
single and fixed multi-scale Difference-of-Averages responses.  It never opens
selection, sealed-audit, competition-test, or leaderboard artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

DEFAULT_SCALES = ((3, 7), (3, 9), (3, 11), (5, 9), (5, 13), (7, 15))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_volume(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = np.quantile(values, (0.001, 0.999))
    if not np.isfinite(low + high) or high <= low:
        raise ValueError("volume has no finite robust intensity range")
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def difference_of_averages(
    values: np.ndarray, inner: int, outer: int
) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("values must be a 3D volume")
    if inner <= 0 or outer <= inner or inner % 2 != 1 or outer % 2 != 1:
        raise ValueError("scales must be positive odd values with inner < outer")
    return uniform_filter(values, size=inner, mode="nearest") - uniform_filter(
        values, size=outer, mode="nearest"
    )


def standardized_response(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    absolute = np.abs(values - median)
    scale = float(np.median(absolute)) * 1.4826
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(values.std()), 1e-6)
    return (values - median) / scale


def peak_coordinates(response: np.ndarray, *, maximum_predictions: int) -> np.ndarray:
    if response.ndim != 3 or maximum_predictions <= 0:
        raise ValueError("response must be 3D and maximum_predictions positive")
    local = response == maximum_filter(response, size=3, mode="nearest")
    coordinates = np.argwhere(local)
    scores = response[local]
    order = np.argsort(-scores, kind="stable")[:maximum_predictions]
    return coordinates[order].astype(np.float32, copy=False)


def point_distances(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    search_radius: float = 6.0,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 3)
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 3)
    if not len(predicted):
        return np.full(len(truth), search_radius, dtype=np.float32)
    distances = np.linalg.norm(truth[:, None] - predicted[None], axis=2).min(axis=1)
    return np.minimum(distances, float(search_radius)).astype(np.float32)


def center_points(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.ndim != 2 or nodes.shape[1] < 4:
        raise ValueError("nodes must have columns (t,z,y,x,...)")
    points = nodes[nodes[:, 0].astype(np.int64) == 1, 1:4].copy()
    points[:, 1:] /= 4.0
    if not len(points) or np.any(points < 0.0) or np.any(points >= 64.0):
        raise ValueError("center-frame points are empty or outside pooled crop")
    return points


def embryo_from_member_name(name: str) -> str:
    """Return the allowed embryo prefix from one optimization member."""

    embryo = Path(name).stem.split("_", 1)[0]
    if embryo not in {"44b6", "6bba"}:
        raise ValueError(f"unexpected optimization embryo: {embryo}")
    return embryo


def response_maps(
    volumes: np.ndarray,
    scales: tuple[tuple[int, int], ...],
) -> dict[str, np.ndarray]:
    volumes = np.asarray(volumes)
    if volumes.shape != (3, 64, 64, 64):
        raise ValueError("volumes must have shape (3,64,64,64)")
    normalized = normalize_volume(volumes)
    current = normalized[1]
    temporal_mean = normalized.mean(axis=0)
    responses: dict[str, np.ndarray] = {}
    standardized: dict[str, np.ndarray] = {}
    for inner, outer in scales:
        name = f"avg{inner}-avg{outer}"
        response = 0.5 * (
            difference_of_averages(current, inner, outer)
            + difference_of_averages(temporal_mean, inner, outer)
        )
        responses[name] = response
        standardized[name] = standardized_response(response)
    for count in (2, 3):
        for names in itertools.combinations(standardized, count):
            responses["equal-z:" + "+".join(names)] = np.mean(
                [standardized[name] for name in names], axis=0
            )
    return responses


def summarize(distances: list[float], *, crops: int) -> dict[str, Any]:
    values = np.asarray(distances, dtype=np.float32)
    return {
        "crops": int(crops),
        "points": len(values),
        "top64_recall_at_2_5_voxels": float(np.mean(values <= 2.5)),
        "mean_capped_distance_voxels": float(values.mean()),
        "p90_capped_distance_voxels": float(np.quantile(values, 0.9)),
    }


def evaluate_archive(
    archive: Path,
    *,
    expected_sha256: str,
    scales: tuple[tuple[int, int], ...] = DEFAULT_SCALES,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("expanded replay archive hash changed")
    accumulated: dict[str, list[float]] = {}
    center_standardized: dict[str, list[float]] = {}
    embryo_accumulated: dict[str, dict[str, list[float]]] = {}
    embryo_center_standardized: dict[str, dict[str, list[float]]] = {}
    embryo_crops = {"44b6": 0, "6bba": 0}
    crops = 0
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
                maps = response_maps(payload["volumes"], scales)
            for name, response in maps.items():
                predicted = peak_coordinates(response, maximum_predictions=64)
                distances = point_distances(predicted, truth)
                accumulated.setdefault(name, []).extend(float(x) for x in distances)
                embryo_accumulated.setdefault(name, {}).setdefault(embryo, []).extend(
                    float(x) for x in distances
                )
                if not name.startswith("equal-z:"):
                    centers = np.clip(np.rint(truth), 0, 63).astype(np.int64)
                    standardized = standardized_response(response)
                    center_values = standardized[
                        centers[:, 0], centers[:, 1], centers[:, 2]
                    ]
                    center_standardized.setdefault(name, []).extend(
                        float(value) for value in center_values
                    )
                    embryo_center_standardized.setdefault(name, {}).setdefault(
                        embryo, []
                    ).extend(float(value) for value in center_values)
            embryo_crops[embryo] += 1
            crops += 1
    rows = [
        {"response": name, **summarize(distances, crops=crops)}
        for name, distances in accumulated.items()
    ]
    for row in rows:
        row["embryos"] = {
            embryo: summarize(
                embryo_accumulated[row["response"]][embryo],
                crops=embryo_crops[embryo],
            )
            for embryo in sorted(embryo_crops)
        }
        values = center_standardized.get(row["response"])
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float32)
        row.update(
            {
                "annotated_center_standardized_p10": float(
                    np.quantile(array, 0.1)
                ),
                "annotated_center_standardized_median": float(np.median(array)),
                "annotated_centers_at_or_below_background_median_fraction": float(
                    np.mean(array <= 0.0)
                ),
            }
        )
        for embryo, embryo_values in embryo_center_standardized[
            row["response"]
        ].items():
            embryo_array = np.asarray(embryo_values, dtype=np.float32)
            row["embryos"][embryo].update(
                {
                    "annotated_center_standardized_p10": float(
                        np.quantile(embryo_array, 0.1)
                    ),
                    "annotated_center_standardized_median": float(
                        np.median(embryo_array)
                    ),
                    "annotated_centers_at_or_below_background_median_fraction": float(
                        np.mean(embryo_array <= 0.0)
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            -row["top64_recall_at_2_5_voxels"],
            row["mean_capped_distance_voxels"],
            row["response"],
        )
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "diagnostic": "expanded_real_optimization_only_multiscale_blob_prior",
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
    parser = argparse.ArgumentParser()
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
    print(json.dumps(report["rows"][:12], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
