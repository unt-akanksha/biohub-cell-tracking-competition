#!/usr/bin/env python
"""Screen motion-tolerant temporal cell evidence on optimization crops only.

The center-frame response remains the anchor.  Adjacent-frame local-SNR
responses may support it within a fixed one- or two-voxel neighborhood, which
allows ordinary inter-frame motion without rewarding a peak that exists only
outside the frame being localized.  This is an architecture diagnostic: it
never reads selection, sealed-audit, competition-test, or leaderboard data.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from research.peak_rank_detection.diagnose_local_snr_prior import local_snr_band
from research.peak_rank_detection.diagnose_multiscale_blob_prior import (
    center_points,
    embryo_from_member_name,
    normalize_volume,
    peak_coordinates,
    point_distances,
    sha256_file,
    standardized_response,
    summarize,
)


PRIOR_LOCAL_SNR_BANDS = ((3, 9), (3, 11), (5, 13))
DEPLOYED_LOCAL_SNR_BANDS = ((3, 9), (5, 13), (7, 15))
MOTION_RADII = (1, 2)


def per_frame_local_snr(
    volumes: np.ndarray,
    bands: tuple[tuple[int, int], ...] = PRIOR_LOCAL_SNR_BANDS,
) -> np.ndarray:
    """Return one equally weighted multiscale local-SNR field per frame."""

    volumes = np.asarray(volumes)
    if volumes.shape != (3, 64, 64, 64):
        raise ValueError("volumes must have shape (3,64,64,64)")
    normalized = normalize_volume(volumes)
    fields = []
    for frame in normalized:
        responses = [
            standardized_response(local_snr_band(frame, inner, outer))
            for inner, outer in bands
        ]
        fields.append(np.mean(responses, axis=0, dtype=np.float32))
    return np.stack(fields).astype(np.float32, copy=False)


def reduced_local_snr_control(
    volumes: np.ndarray,
    bands: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Reproduce the current-plus-temporal-minimum diagnostic efficiently."""

    volumes = np.asarray(volumes)
    if volumes.shape != (3, 64, 64, 64):
        raise ValueError("volumes must have shape (3,64,64,64)")
    normalized = normalize_volume(volumes)
    current = normalized[1]
    stable = normalized.min(axis=0)
    responses = []
    for inner, outer in bands:
        responses.extend(
            (
                standardized_response(local_snr_band(current, inner, outer)),
                standardized_response(local_snr_band(stable, inner, outer)),
            )
        )
    return np.mean(responses, axis=0, dtype=np.float32)


def motion_supported_maps(volumes: np.ndarray) -> dict[str, np.ndarray]:
    """Build fixed, center-anchored temporal support responses."""

    frame_fields = per_frame_local_snr(volumes)
    previous, current, following = frame_fields
    maps: dict[str, np.ndarray] = {
        "current:per-frame-local-snr": current,
        "mean:per-frame-local-snr": frame_fields.mean(axis=0),
        "minimum:per-frame-local-snr": frame_fields.min(axis=0),
    }
    for radius in MOTION_RADII:
        size = 2 * radius + 1
        previous_support = maximum_filter(previous, size=size, mode="nearest")
        following_support = maximum_filter(following, size=size, mode="nearest")
        supported = np.stack((current, previous_support, following_support))
        maps[f"motion-mean-r{radius}:per-frame-local-snr"] = supported.mean(axis=0)
        maps[f"motion-minimum-r{radius}:per-frame-local-snr"] = supported.min(axis=0)

    # Compare the exact scales that justified V23 with the scales currently
    # staged in its implementation before any held-out gate is opened.
    maps["control:minimum-reduced-local-snr-prior-bands"] = (
        reduced_local_snr_control(volumes, PRIOR_LOCAL_SNR_BANDS)
    )
    maps["control:minimum-reduced-local-snr-deployed-bands"] = (
        reduced_local_snr_control(volumes, DEPLOYED_LOCAL_SNR_BANDS)
    )
    return {name: np.asarray(values, dtype=np.float32) for name, values in maps.items()}


def score_archive_member(
    item: tuple[str, bytes],
) -> tuple[str, dict[str, np.ndarray]]:
    """Score one already role-filtered member without shared mutable state."""

    name, content = item
    embryo = embryo_from_member_name(name)
    with np.load(io.BytesIO(content), allow_pickle=False) as payload:
        truth = center_points(payload["nodes"])
        maps = motion_supported_maps(payload["volumes"])
    return embryo, {
        map_name: point_distances(
            peak_coordinates(response, maximum_predictions=64), truth
        )
        for map_name, response in maps.items()
    }


def evaluate_archive(
    archive: Path, *, expected_sha256: str, workers: int = 1
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
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
        payloads: list[tuple[str, bytes]] = []
        for member in optimization:
            if "/selection/" in member.name or "/sealed_audit/" in member.name:
                raise ValueError("diagnostic attempted to cross the optimization boundary")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member: {member.name}")
            payloads.append((member.name, extracted.read()))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scored_members = executor.map(score_archive_member, payloads)
        for embryo, scored in scored_members:
            for name, distances in scored.items():
                accumulated.setdefault(name, []).extend(float(value) for value in distances)
                by_embryo.setdefault(name, {}).setdefault(embryo, []).extend(
                    float(value) for value in distances
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
        "diagnostic": "optimization_only_motion_supported_temporal_local_snr_prior",
        "archive_sha256": actual_sha256,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "selection_data_read": False,
        "sealed_audit_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "maximum_predictions_per_crop": 64,
        "workers": workers,
        "prior_local_snr_bands": [list(band) for band in PRIOR_LOCAL_SNR_BANDS],
        "deployed_local_snr_bands": [
            list(band) for band in DEPLOYED_LOCAL_SNR_BANDS
        ],
        "motion_radii_voxels": list(MOTION_RADII),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_archive(
        args.archive.resolve(),
        expected_sha256=args.expected_sha256,
        workers=args.workers,
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
