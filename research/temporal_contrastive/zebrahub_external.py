"""Selective, provenance-bound ZebraHub temporal patch extraction.

The public ZebraHub movies are too large to mirror for one competition model.
This module reads deterministic OME-Zarr time blocks, aligns their dense lineage
CSV rows, and emits the same physical 17-cubed temporal patches used by the
project-authored Biohub appearance model.  It never reads competition test data
or public competition predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from numcodecs import get_codec

try:
    from patch_model import sample_physical_patches
    from transition_context import (
        CANDIDATE_CONTEXT_WIDTH,
        candidate_transition_features,
        estimate_transition_context,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.patch_model import sample_physical_patches
    from research.temporal_contrastive.transition_context import (
        CANDIDATE_CONTEXT_WIDTH,
        candidate_transition_features,
        estimate_transition_context,
    )


PUBLIC_ROOT = (
    "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective"
)
ORGANIZER_AUTHORIZATION = (
    "https://www.kaggle.com/competitions/"
    "biohub-cell-tracking-during-development/discussion/734330"
)
EXPECTED_LEVEL0_SPACING_UM = (1.24, 0.439, 0.439)
EXPECTED_LEVEL1_SPACING_UM = (2.48, 0.878, 0.878)

SOURCE_SPECS = {
    "ZSNS004": {
        "role": "external_pretraining",
        "frames": 600,
        "tracks_bytes": 377_693_868,
    },
    "ZSNS005": {
        "role": "external_validation",
        "frames": 600,
        "tracks_bytes": 341_483_571,
    },
}
SAMPLING_POLICY = "seeded_local_neighborhood_balanced_v2"
DIVISION_QUOTA_FRACTION = 0.125
MAXIMUM_DIVISION_SOURCE_FRACTION = 0.25


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "public.czbiohub.org":
        raise ValueError("ZebraHub source must use the public CZ Biohub HTTPS host")
    required = "/royerlab/zebrahub/imaging/single-objective/"
    if not parsed.path.startswith(required):
        raise ValueError("URL is outside the public single-objective inventory")
    if parsed.query or parsed.fragment:
        raise ValueError("source URLs cannot contain a query or fragment")
    return url


def _positive_spacing(values: Sequence[float]) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in values)
    if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive finite values")
    return spacing  # type: ignore[return-value]


@dataclass(frozen=True)
class DenseTransition:
    source_ids: np.ndarray
    target_ids: np.ndarray
    source_coords_level0: np.ndarray
    target_coords_level0: np.ndarray
    source_coords_um: np.ndarray
    target_coords_um: np.ndarray
    candidate_mask: np.ndarray
    positive_mask: np.ndarray
    division_target: np.ndarray

    def validate(self) -> None:
        rows, columns = self.candidate_mask.shape
        if self.positive_mask.shape != (rows, columns):
            raise ValueError("positive and candidate masks differ")
        if rows != len(self.source_ids) or columns != len(self.target_ids):
            raise ValueError("transition identifiers do not align with masks")
        if self.source_coords_level0.shape != (rows, 3):
            raise ValueError("source coordinates do not align")
        if self.target_coords_level0.shape != (columns, 3):
            raise ValueError("target coordinates do not align")
        if self.source_coords_um.shape != (rows, 3):
            raise ValueError("source physical coordinates do not align")
        if self.target_coords_um.shape != (columns, 3):
            raise ValueError("target physical coordinates do not align")
        if self.division_target.shape != (rows,):
            raise ValueError("division targets do not align")
        if self.candidate_mask.dtype != np.bool_ or self.positive_mask.dtype != np.bool_:
            raise ValueError("transition masks must be boolean")
        if np.any(self.positive_mask & ~self.candidate_mask):
            raise ValueError("a positive edge is outside the candidate set")
        if np.any(self.candidate_mask.sum(axis=1) <= self.positive_mask.sum(axis=1)):
            raise ValueError("every selected source requires a hard negative")
        if not np.array_equal(
            self.division_target, (self.positive_mask.sum(axis=1) >= 2).astype(np.float32)
        ):
            raise ValueError("division targets changed")
        tensors = (
            self.source_coords_level0,
            self.target_coords_level0,
            self.source_coords_um,
            self.target_coords_um,
            self.division_target,
        )
        if any(not np.isfinite(value).all() for value in tensors):
            raise ValueError("transition contains non-finite values")


def select_dense_transition(
    source_ids: np.ndarray,
    source_coords_level0: np.ndarray,
    target_ids: np.ndarray,
    target_coords_level0: np.ndarray,
    target_parent_ids: np.ndarray,
    *,
    voxel_size_level0_um: Sequence[float] = EXPECTED_LEVEL0_SPACING_UM,
    radius_um: float = 32.0,
    max_sources: int = 176,
    max_targets: int = 176,
    seed: int = 0,
) -> DenseTransition:
    """Select a bounded dense transition while retaining all chosen daughters."""

    source_ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    target_ids = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    parents = np.asarray(target_parent_ids, dtype=np.int64).reshape(-1)
    source_coords = np.asarray(source_coords_level0, dtype=np.float32)
    target_coords = np.asarray(target_coords_level0, dtype=np.float32)
    spacing = np.asarray(_positive_spacing(voxel_size_level0_um), dtype=np.float32)
    if source_coords.shape != (len(source_ids), 3):
        raise ValueError("source coordinates must have shape (N, 3)")
    if target_coords.shape != (len(target_ids), 3):
        raise ValueError("target coordinates must have shape (M, 3)")
    if len(parents) != len(target_ids):
        raise ValueError("one parent identifier is required per target")
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(target_ids)) != len(target_ids):
        raise ValueError("node identifiers must be unique within each frame")
    if not np.isfinite(source_coords).all() or not np.isfinite(target_coords).all():
        raise ValueError("coordinates must be finite")
    if not math.isfinite(float(radius_um)) or radius_um <= 0:
        raise ValueError("candidate radius must be positive and finite")
    if max_sources <= 0 or max_targets <= 1:
        raise ValueError("transition bounds are too small")

    source_lookup = {int(node): row for row, node in enumerate(source_ids)}
    positives_by_source: dict[int, list[int]] = {}
    for target_row, parent in enumerate(parents):
        source_row = source_lookup.get(int(parent))
        if source_row is not None:
            positives_by_source.setdefault(source_row, []).append(target_row)
    if not positives_by_source:
        raise ValueError("selected frames contain no consecutive lineage edges")

    source_um_all = source_coords * spacing[None]
    target_um_all = target_coords * spacing[None]
    eligible = np.asarray(
        [
            row
            for row in sorted(positives_by_source)
            if all(
                np.linalg.norm(source_um_all[row] - target_um_all[target_row])
                <= float(radius_um)
                for target_row in positives_by_source[row]
            )
        ],
        dtype=np.int64,
    )
    if not len(eligible):
        raise ValueError(
            "candidate radius omitted selected ZebraHub links; "
            "no source retained all lineage edges"
        )
    rng = np.random.default_rng(int(seed))
    anchor = int(eligible[int(rng.integers(0, len(eligible)))])
    anchor_distances = np.linalg.norm(
        source_um_all - source_um_all[anchor][None], axis=1
    )

    def nearest(rows: np.ndarray) -> np.ndarray:
        if not len(rows):
            return rows
        return rows[np.lexsort((source_ids[rows], anchor_distances[rows]))]

    division_mask = np.asarray(
        [len(positives_by_source[int(row)]) >= 2 for row in eligible], dtype=bool
    )
    divisions = nearest(eligible[division_mask])
    ordinary = nearest(eligible[~division_mask])
    division_quota = min(
        len(divisions), max(1, int(round(max_sources * DIVISION_QUOTA_FRACTION)))
    )
    selected_pool = np.concatenate(
        (divisions[:division_quota], ordinary[: max_sources - division_quota])
    )
    source_order = nearest(selected_pool)
    selected_sources: list[int] = []
    required_targets: set[int] = set()
    for source_row in source_order.tolist():
        proposed = required_targets | set(positives_by_source[source_row])
        if len(proposed) > max_targets:
            continue
        selected_sources.append(source_row)
        required_targets = proposed
        if len(selected_sources) == max_sources:
            break
    if not selected_sources:
        raise ValueError("transition bounds cannot retain one labeled source")

    selected_source_rows = np.asarray(selected_sources, dtype=np.int64)
    source_um = source_um_all[selected_source_rows]
    distances = np.linalg.norm(
        source_um[:, None, :] - target_um_all[None, :, :], axis=2
    )
    full_candidates = distances <= float(radius_um)
    full_positives = np.zeros_like(full_candidates, dtype=bool)
    for local_source, global_source in enumerate(selected_sources):
        full_positives[local_source, positives_by_source[global_source]] = True
    if np.any(full_positives & ~full_candidates):
        missed = distances[full_positives & ~full_candidates]
        raise ValueError(
            "candidate radius omitted selected ZebraHub links; "
            f"largest positive distance={float(missed.max()):.6f} um"
        )

    required = np.flatnonzero(full_positives.any(axis=0))
    available = np.flatnonzero(full_candidates.any(axis=0))
    optional = np.setdiff1d(available, required, assume_unique=True)
    if len(optional):
        nearest = distances[:, optional].min(axis=0)
        optional_order = np.lexsort((target_ids[optional], nearest))
        optional = optional[optional_order]
    capacity = max_targets - len(required)
    selected_targets = np.sort(
        np.concatenate((required, optional[: max(0, capacity)]))
    )
    candidates = full_candidates[:, selected_targets]
    positives = full_positives[:, selected_targets]
    usable = candidates.sum(axis=1) > positives.sum(axis=1)
    if not np.any(usable):
        raise ValueError("selected ZebraHub sources have no hard negatives")
    selected_source_rows = selected_source_rows[usable]
    candidates = candidates[usable]
    positives = positives[usable]
    division_rows = np.flatnonzero(positives.sum(axis=1) >= 2)
    ordinary_count = int(np.sum(positives.sum(axis=1) == 1))
    allowed_divisions = max(
        1,
        int(
            math.floor(
                ordinary_count
                * MAXIMUM_DIVISION_SOURCE_FRACTION
                / (1.0 - MAXIMUM_DIVISION_SOURCE_FRACTION)
            )
        ),
    )
    if len(division_rows) > allowed_divisions:
        keep = np.ones(len(selected_source_rows), dtype=bool)
        keep[division_rows[allowed_divisions:]] = False
        selected_source_rows = selected_source_rows[keep]
        candidates = candidates[keep]
        positives = positives[keep]
    if not np.any(positives.sum(axis=1) == 1):
        raise ValueError("balanced ZebraHub selection contains no ordinary links")

    result = DenseTransition(
        source_ids=source_ids[selected_source_rows].copy(),
        target_ids=target_ids[selected_targets].copy(),
        source_coords_level0=source_coords[selected_source_rows].copy(),
        target_coords_level0=target_coords[selected_targets].copy(),
        source_coords_um=(source_coords[selected_source_rows] * spacing[None]).copy(),
        target_coords_um=(target_coords[selected_targets] * spacing[None]).copy(),
        candidate_mask=candidates.copy(),
        positive_mask=positives.copy(),
        division_target=(positives.sum(axis=1) >= 2).astype(np.float32),
    )
    result.validate()
    return result


def filtered_csv_frames(
    path: Path, frame_times: Iterable[int]
) -> dict[int, dict[str, np.ndarray]]:
    """Read only requested frames from a dense, track-major ZebraHub CSV."""

    requested = {int(value) for value in frame_times}
    if not requested or min(requested) < 1:
        raise ValueError("CSV frame times must be positive")
    rows: dict[int, dict[str, list[Any]]] = {
        timepoint: {"id": [], "parent_id": [], "coords": []}
        for timepoint in requested
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"t", "z", "y", "x", "id", "parent_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("ZebraHub lineage CSV schema changed")
        for record in reader:
            timepoint = int(record["t"])
            if timepoint not in requested:
                continue
            row = rows[timepoint]
            row["id"].append(int(record["id"]))
            row["parent_id"].append(int(record["parent_id"]))
            row["coords"].append(
                (float(record["z"]), float(record["y"]), float(record["x"]))
            )
    result: dict[int, dict[str, np.ndarray]] = {}
    for timepoint, values in rows.items():
        if not values["id"]:
            raise ValueError(f"ZebraHub CSV has no nodes at frame {timepoint}")
        result[timepoint] = {
            "id": np.asarray(values["id"], dtype=np.int64),
            "parent_id": np.asarray(values["parent_id"], dtype=np.int64),
            "coords": np.asarray(values["coords"], dtype=np.float32),
        }
    return result


class PublicZarrV2Array:
    """Minimal cached reader for one public, single-channel Zarr v2 array."""

    def __init__(self, base_url: str, cache_dir: Path, *, timeout: int = 120) -> None:
        self.base_url = validate_public_url(base_url.rstrip("/") + "/")[:-1]
        self.cache_dir = cache_dir
        self.timeout = int(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_bytes = self._get(".zarray", cache_name="zarray.json")
        self.metadata_sha256 = sha256_bytes(metadata_bytes)
        self.metadata = json.loads(metadata_bytes.decode("utf-8"))
        if self.metadata.get("zarr_format") != 2:
            raise ValueError("only Zarr v2 arrays are supported")
        if self.metadata.get("order") != "C" or self.metadata.get("dimension_separator") != "/":
            raise ValueError("unexpected ZebraHub Zarr storage layout")
        self.shape = tuple(int(value) for value in self.metadata["shape"])
        self.chunks = tuple(int(value) for value in self.metadata["chunks"])
        if len(self.shape) != 5 or self.shape[1] != 1 or len(self.chunks) != 5:
            raise ValueError("expected a five-dimensional single-channel movie")
        self.dtype = np.dtype(self.metadata["dtype"])
        self.codec = get_codec(self.metadata["compressor"])
        self.downloads: list[dict[str, Any]] = []

    def _get(self, relative: str, *, cache_name: str) -> bytes:
        target = self.cache_dir / cache_name
        if target.is_file():
            return target.read_bytes()
        url = f"{self.base_url}/{relative}"
        validate_public_url(url)
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.content
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
        return payload

    def read_frame(self, time_index: int) -> np.ndarray:
        if not 0 <= int(time_index) < self.shape[0]:
            raise ValueError("time index is outside the Zarr movie")
        spatial_shape = self.shape[2:]
        spatial_chunks = self.chunks[2:]
        frame = np.full(spatial_shape, self.metadata.get("fill_value", 0), dtype=self.dtype)
        grid = tuple(math.ceil(size / chunk) for size, chunk in zip(spatial_shape, spatial_chunks, strict=True))
        for z_index in range(grid[0]):
            for y_index in range(grid[1]):
                for x_index in range(grid[2]):
                    relative = f"{time_index}/0/{z_index}/{y_index}/{x_index}"
                    cache_name = f"t{time_index:04d}-z{z_index}-y{y_index}-x{x_index}.blosc"
                    payload = self._get(relative, cache_name=cache_name)
                    decoded = self.codec.decode(payload)
                    expected = int(np.prod(self.chunks))
                    values = np.frombuffer(decoded, dtype=self.dtype)
                    if values.size != expected:
                        raise ValueError("decoded Zarr chunk has an unexpected size")
                    chunk = values.reshape(self.chunks)[0, 0]
                    starts = (
                        z_index * spatial_chunks[0],
                        y_index * spatial_chunks[1],
                        x_index * spatial_chunks[2],
                    )
                    stops = tuple(
                        min(start + chunk_size, full_size)
                        for start, chunk_size, full_size in zip(
                            starts, spatial_chunks, spatial_shape, strict=True
                        )
                    )
                    valid = tuple(slice(0, stop - start) for start, stop in zip(starts, stops, strict=True))
                    destination = tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))
                    frame[destination] = chunk[valid]
                    self.downloads.append(
                        {
                            "url": f"{self.base_url}/{relative}",
                            "bytes": len(payload),
                            "sha256": sha256_bytes(payload),
                        }
                    )
        return frame


def download_public_file(url: str, target: Path, *, expected_bytes: int) -> dict[str, Any]:
    validate_public_url(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_suffix(target.suffix + ".partial")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        stream.write(block)
        os.replace(temporary, target)
    size = target.stat().st_size
    if size != int(expected_bytes):
        raise ValueError(f"public file size changed: expected {expected_bytes}, saw {size}")
    return {"url": url, "bytes": size, "sha256": sha256_file(target)}


def _one_channel_patches(
    volume: np.ndarray,
    centers_level0: np.ndarray,
    *,
    downsample: float,
    voxel_size_level1_um: Sequence[float],
) -> np.ndarray:
    centers = np.asarray(centers_level0, dtype=np.float32) / float(downsample)
    patches = sample_physical_patches(
        volume,
        centers,
        voxel_size_zyx_um=voxel_size_level1_um,
        output_shape_zyx=(17, 17, 17),
        half_extent_zyx_um=(8.0, 8.0, 8.0),
        chunk_size=32,
    )
    result = patches[:, 0].cpu().numpy().astype(np.float16)
    if result.shape != (len(centers), 17, 17, 17) or not np.isfinite(result).all():
        raise RuntimeError("ZebraHub patch extraction produced invalid values")
    return result


def build_temporal_patch_shard(
    source_name: str,
    *,
    csv_timepoint: int,
    cache_root: Path,
    output_path: Path,
    seed: int = 44004,
    max_sources: int = 64,
    max_targets: int = 96,
    radius_um: float = 32.0,
    preloaded_rows: dict[int, dict[str, np.ndarray]] | None = None,
    tracks_record: dict[str, Any] | None = None,
    zarr_reader: PublicZarrV2Array | None = None,
) -> dict[str, Any]:
    """Build one deterministic four-frame ZebraHub temporal-patch shard."""

    if source_name not in SOURCE_SPECS:
        raise ValueError(f"unsupported or unfrozen ZebraHub source: {source_name}")
    spec = SOURCE_SPECS[source_name]
    timepoint = int(csv_timepoint)
    if not 2 <= timepoint <= int(spec["frames"]) - 2:
        raise ValueError("timepoint requires t-1 through t+2 image context")
    source_root = cache_root / source_name
    tracks_url = f"{PUBLIC_ROOT}/{source_name}_tracks.csv"
    tracks_path = source_root / f"{source_name}_tracks.csv"
    if tracks_record is None:
        tracks_record = download_public_file(
            tracks_url, tracks_path, expected_bytes=int(spec["tracks_bytes"])
        )
    elif not (
        tracks_record.get("url") == tracks_url
        and tracks_record.get("bytes") == int(spec["tracks_bytes"])
        and tracks_record.get("sha256") == sha256_file(tracks_path)
    ):
        raise ValueError("preloaded ZebraHub tracks evidence changed")
    if preloaded_rows is None:
        rows = filtered_csv_frames(tracks_path, (timepoint, timepoint + 1))
    else:
        try:
            rows = {
                timepoint: preloaded_rows[timepoint],
                timepoint + 1: preloaded_rows[timepoint + 1],
            }
        except KeyError as error:
            raise ValueError("preloaded ZebraHub rows omit the transition") from error
    source_rows = rows[timepoint]
    target_rows = rows[timepoint + 1]
    transition = select_dense_transition(
        source_rows["id"],
        source_rows["coords"],
        target_rows["id"],
        target_rows["coords"],
        target_rows["parent_id"],
        radius_um=radius_um,
        max_sources=max_sources,
        max_targets=max_targets,
        seed=seed,
    )

    level_url = f"{PUBLIC_ROOT}/{source_name}.ome.zarr/1"
    if zarr_reader is None:
        zarr_reader = PublicZarrV2Array(level_url, source_root / "level1")
    elif zarr_reader.base_url != level_url:
        raise ValueError("preloaded ZebraHub image source changed")
    if zarr_reader.shape[0] != int(spec["frames"]):
        raise ValueError("ZebraHub frame count changed")
    download_start = len(zarr_reader.downloads)
    source_patches = np.empty((len(transition.source_ids), 3, 17, 17, 17), dtype=np.float16)
    target_patches = np.empty((len(transition.target_ids), 3, 17, 17, 17), dtype=np.float16)
    required_csv_times = (timepoint - 1, timepoint, timepoint + 1, timepoint + 2)
    transition_frames: dict[int, np.ndarray] = {}
    for csv_frame in required_csv_times:
        frame = zarr_reader.read_frame(csv_frame - 1)
        if csv_frame in {timepoint, timepoint + 1}:
            transition_frames[csv_frame] = frame
        source_channel = csv_frame - (timepoint - 1)
        if 0 <= source_channel < 3:
            source_patches[:, source_channel] = _one_channel_patches(
                frame,
                transition.source_coords_level0,
                downsample=2.0,
                voxel_size_level1_um=EXPECTED_LEVEL1_SPACING_UM,
            )
        target_channel = csv_frame - timepoint
        if 0 <= target_channel < 3:
            target_patches[:, target_channel] = _one_channel_patches(
                frame,
                transition.target_coords_level0,
                downsample=2.0,
                voxel_size_level1_um=EXPECTED_LEVEL1_SPACING_UM,
            )
        if csv_frame not in transition_frames:
            del frame
    if not np.isfinite(source_patches).all() or not np.isfinite(target_patches).all():
        raise RuntimeError("temporal patch shard contains non-finite values")
    if set(transition_frames) != {timepoint, timepoint + 1}:
        raise RuntimeError("central ZebraHub transition frames were not retained")
    transition_context = estimate_transition_context(
        transition_frames[timepoint],
        transition_frames[timepoint + 1],
        voxel_size_zyx_um=EXPECTED_LEVEL1_SPACING_UM,
    )
    candidate_context = candidate_transition_features(
        transition.source_coords_um,
        transition.target_coords_um,
        transition.candidate_mask,
        transition_context,
        candidate_radius_um=radius_um,
    )
    if candidate_context.shape != (
        len(transition.source_ids),
        len(transition.target_ids),
        CANDIDATE_CONTEXT_WIDTH,
    ):
        raise RuntimeError("ZebraHub candidate context shape changed")
    del transition_frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_patches=source_patches,
        target_patches=target_patches,
        source_ids=transition.source_ids,
        target_ids=transition.target_ids,
        source_coords_um=transition.source_coords_um,
        target_coords_um=transition.target_coords_um,
        candidate_mask=transition.candidate_mask,
        positive_mask=transition.positive_mask,
        division_target=transition.division_target,
        transition_context=transition_context.feature_vector(
            candidate_radius_um=radius_um
        ),
        candidate_context=candidate_context,
    )
    shard_hash = sha256_file(output_path)
    unique_downloads = {
        row["url"]: row for row in zarr_reader.downloads[download_start:]
    }
    manifest = {
        "schema_version": 1,
        "source": source_name,
        "source_role": spec["role"],
        "csv_timepoint": timepoint,
        "seed": int(seed),
        "organizer_authorization": ORGANIZER_AUTHORIZATION,
        "organizer_declared_test_overlap": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
        "level0_spacing_um": list(EXPECTED_LEVEL0_SPACING_UM),
        "level1_spacing_um": list(EXPECTED_LEVEL1_SPACING_UM),
        "temporal_offsets": [-1, 0, 1],
        "patch_shape": [17, 17, 17],
        "patch_half_extent_um": [8.0, 8.0, 8.0],
        "candidate_radius_um": float(radius_um),
        "sampling_policy": SAMPLING_POLICY,
        "division_quota_fraction": DIVISION_QUOTA_FRACTION,
        "maximum_division_source_fraction": MAXIMUM_DIVISION_SOURCE_FRACTION,
        "source_nodes": len(transition.source_ids),
        "target_nodes": len(transition.target_ids),
        "positive_edges": int(transition.positive_mask.sum()),
        "division_sources": int(transition.division_target.sum()),
        "candidate_edges": int(transition.candidate_mask.sum()),
        "transition_context": transition_context.to_dict(),
        "candidate_context_width": CANDIDATE_CONTEXT_WIDTH,
        "tracks": tracks_record,
        "zarr_level_url": level_url,
        "zarr_metadata_sha256": zarr_reader.metadata_sha256,
        "zarr_chunks": list(unique_downloads.values()),
        "shard": {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": shard_hash,
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
