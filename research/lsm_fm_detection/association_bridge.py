"""Bridge independently trained LSM-FM detections into the official linker.

The official Biohub baseline's association network consumes integer detector
coordinates on the ``(1, 4, 4)`` input grid.  This module substitutes only
that detector callback, then restores the LSM-FM detector's sub-voxel
coordinates after edge inference.  The edge model and graph construction stay
unchanged, while detector identities and coordinates come exclusively from
our learned checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

try:
    from density_calibration import read_estimated_node_count, uniform_frame_indices
    from inference import predict_probability_batch
    from localization_refinement import refine_peaks_log_quadratic, refine_peaks_weighted
    from pu_targets import extract_local_peaks
    from train_spatialdino_pu_detector import normalize_spatialdino_frame
except ModuleNotFoundError:
    from research.density_calibration import read_estimated_node_count, uniform_frame_indices
    from research.lsm_fm_detection.localization_refinement import (
        refine_peaks_log_quadratic,
        refine_peaks_weighted,
    )
    from research.spatialdino_detection.inference import predict_probability_batch
    from research.spatialdino_detection.train_pu_detector import (
        normalize_spatialdino_frame,
    )
    from research.spotiflow_biohub.pu_targets import extract_local_peaks


LOW_PROBABILITY_THRESHOLD = 0.02
SPATIAL_DOWNSAMPLE = np.asarray((1.0, 4.0, 4.0), dtype=np.float32)
ASSOCIATION_COORDINATE_MODES = frozenset({"rounded", "subvoxel"})


@dataclass(frozen=True)
class DetectorCandidate:
    """Frozen probability blend and localization rule selected off leaderboard."""

    name: str
    feature24_weight: float
    feature36_weight: float
    refinement: str = "centroid"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("candidate name must be nonempty")
        if self.feature24_weight < 0 or self.feature36_weight < 0:
            raise ValueError("detector weights must be nonnegative")
        if self.feature24_weight + self.feature36_weight <= 0:
            raise ValueError("at least one detector weight must be positive")
        if self.refinement not in {"centroid", "log_quadratic"}:
            raise ValueError("refinement must be centroid or log_quadratic")


@dataclass(frozen=True)
class FrameDetections:
    frame: int
    points_input: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_input, dtype=np.float32)
        scores = np.asarray(self.confidence, dtype=np.float32)
        if self.frame < 0:
            raise ValueError("frame must be nonnegative")
        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_input must be a finite (N, 3) array")
        if scores.shape != (len(points),) or not np.isfinite(scores).all():
            raise ValueError("confidence must be a finite vector aligned with points")
        if scores.size and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
            raise ValueError("confidence must lie in [0, 1]")
        object.__setattr__(self, "points_input", points)
        object.__setattr__(self, "confidence", scores)


@dataclass(frozen=True)
class ExternalDetectionCache:
    """Thresholded movie detections in association and output coordinates."""

    candidate: DetectorCandidate
    frames: tuple[FrameDetections, ...]
    threshold: float
    projected_node_count: float
    estimated_node_count: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        if not np.isfinite(self.estimated_node_count) or self.estimated_node_count <= 0:
            raise ValueError("estimated node count must be finite and positive")
        ids = tuple(row.frame for row in self.frames)
        if ids != tuple(range(len(self.frames))):
            raise ValueError("detections must cover every frame once in order")

    def selected_points(self, frame: int) -> np.ndarray:
        row = self.frames[int(frame)]
        if row.frame != frame:
            raise RuntimeError("frame lookup is inconsistent")
        return row.points_input[row.confidence > self.threshold]

    def association_coords(
        self, frame: int, *, mode: str = "rounded"
    ) -> np.ndarray:
        """Return official-linker coordinates on its pooled input grid.

        ``rounded`` preserves the organizer baseline contract. ``subvoxel``
        keeps the detector's bounded offsets for positional and pairwise edge
        scoring; feature-map lookup is still rounded separately so the frozen
        linker sees the same kind of sampled image features it was trained on.
        """

        points = self.selected_points(frame)
        if mode not in ASSOCIATION_COORDINATE_MODES:
            raise ValueError(f"unsupported association coordinate mode: {mode}")
        if not len(points):
            dtype = np.int16 if mode == "rounded" else np.float32
            return np.empty((0, 4), dtype=dtype)
        spatial = (
            np.rint(points).astype(np.int64)
            if mode == "rounded"
            else np.asarray(points, dtype=np.float32)
        )
        if np.any(spatial < 0) or np.any(spatial > np.iinfo(np.int16).max):
            raise ValueError("association coordinate is outside int16 bounds")
        dtype = np.int16 if mode == "rounded" else np.float32
        times = np.full((len(spatial), 1), int(frame), dtype=dtype)
        return np.concatenate((times, spatial), axis=1).astype(dtype, copy=False)

    def precise_output_coords(self) -> np.ndarray:
        """Return ``[t,z,y,x]`` with sub-voxel positions in original space."""

        rows = []
        for frame in range(len(self.frames)):
            points = self.selected_points(frame) * SPATIAL_DOWNSAMPLE
            if len(points):
                times = np.full((len(points), 1), float(frame), dtype=np.float32)
                rows.append(np.concatenate((times, points), axis=1))
        return np.concatenate(rows).astype(np.float32) if rows else np.empty((0, 4), np.float32)

    def restore_precise_output(self, linked_coords: np.ndarray) -> np.ndarray:
        """Validate node identity/order, then restore precise LSM coordinates."""

        linked = np.asarray(linked_coords)
        precise = self.precise_output_coords()
        if linked.ndim != 2 or linked.shape[1:] != (4,):
            raise ValueError("linked_coords must have shape (N, 4)")
        if len(linked) != len(precise):
            raise RuntimeError("official linker changed the external detector node count")
        if not np.array_equal(linked[:, 0].astype(np.int64), precise[:, 0].astype(np.int64)):
            raise RuntimeError("official linker changed external detector frame ordering")
        return precise

    def manifest(self) -> dict[str, Any]:
        counts = [len(self.selected_points(frame)) for frame in range(len(self.frames))]
        return {
            "candidate": self.candidate.name,
            "feature24_weight": self.candidate.feature24_weight,
            "feature36_weight": self.candidate.feature36_weight,
            "refinement": self.candidate.refinement,
            "threshold": self.threshold,
            "projected_node_count": self.projected_node_count,
            "estimated_node_count": self.estimated_node_count,
            "selected_node_count": int(sum(counts)),
            "frame_counts": counts,
            "spatial_downsample": SPATIAL_DOWNSAMPLE.tolist(),
        }


def combine_probabilities(
    feature24: np.ndarray,
    feature36: np.ndarray,
    candidate: DetectorCandidate,
) -> np.ndarray:
    first = np.asarray(feature24, dtype=np.float32)
    second = np.asarray(feature36, dtype=np.float32)
    if first.ndim != 3 or first.shape != second.shape:
        raise ValueError("feature probabilities must be equal-shape 3D volumes")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("feature probabilities must be finite")
    total = candidate.feature24_weight + candidate.feature36_weight
    return (
        first * candidate.feature24_weight + second * candidate.feature36_weight
    ) / total


def detections_from_probability(
    probability: np.ndarray,
    *,
    frame: int,
    candidate: DetectorCandidate,
) -> FrameDetections:
    peak_set = extract_local_peaks(
        probability,
        threshold=LOW_PROBABILITY_THRESHOLD,
        min_distance_voxels=1,
    )
    if candidate.refinement == "log_quadratic":
        points = refine_peaks_log_quadratic(probability, peak_set.coords)
    else:
        points = refine_peaks_weighted(
            probability,
            peak_set.coords,
            radius=1,
            probability_power=2.0,
        )
    return FrameDetections(frame, points, peak_set.confidence)


def select_density_threshold(
    frames: Sequence[FrameDetections],
    *,
    estimated_node_count: float,
    calibration_frames: int = 12,
) -> tuple[float, float]:
    """Match organizer-provided density using fixed uniform calibration frames."""

    if not frames:
        raise ValueError("frames cannot be empty")
    if not np.isfinite(estimated_node_count) or estimated_node_count <= 0:
        raise ValueError("estimated node count must be finite and positive")
    indices = uniform_frame_indices(len(frames), calibration_frames)
    pooled = np.concatenate([frames[int(index)].confidence for index in indices])
    if not len(pooled):
        return LOW_PROBABILITY_THRESHOLD, 0.0
    desired = int(round(float(estimated_node_count) * len(indices) / len(frames)))
    desired = max(1, min(desired, len(pooled)))
    ordered = np.sort(pooled)[::-1]
    boundary = float(ordered[desired - 1])
    if desired < len(ordered):
        threshold = (boundary + float(ordered[desired])) / 2.0
    else:
        threshold = float(np.nextafter(LOW_PROBABILITY_THRESHOLD, 0.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    selected = sum(
        int(np.count_nonzero(frames[int(index)].confidence > threshold))
        for index in indices
    )
    projected = float(selected) * len(frames) / len(indices)
    return threshold, projected


def build_detection_cache(
    frames: Sequence[FrameDetections],
    *,
    candidate: DetectorCandidate,
    estimated_node_count: float,
    calibration_frames: int = 12,
) -> ExternalDetectionCache:
    ordered = tuple(frames)
    threshold, projected = select_density_threshold(
        ordered,
        estimated_node_count=estimated_node_count,
        calibration_frames=calibration_frames,
    )
    return ExternalDetectionCache(
        candidate=candidate,
        frames=ordered,
        threshold=threshold,
        projected_node_count=projected,
        estimated_node_count=float(estimated_node_count),
    )


def estimated_count_for_movie(sample_path: Path) -> float:
    estimate = read_estimated_node_count(sample_path)
    if estimate is None:
        sibling_geff = sample_path.with_suffix(".geff")
        estimate = read_estimated_node_count(sibling_geff)
    if estimate is None:
        raise ValueError(f"missing estimated_number_of_nodes for {sample_path}")
    return estimate


def predict_movie_detections(
    models: Mapping[str, Any],
    sample_path: Path,
    *,
    candidate: DetectorCandidate,
    device: Any,
    batch_size: int = 1,
    calibration_frames: int = 12,
) -> ExternalDetectionCache:
    """Run the selected LSM-FM detector candidate over one complete movie."""

    import torch
    import zarr

    if set(models) != {"feature24", "feature36"}:
        raise ValueError("models must contain exactly feature24 and feature36")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    array = zarr.open_group(str(sample_path), mode="r")["0"]
    rows = []
    for start in range(0, int(array.shape[0]), batch_size):
        indices = list(range(start, min(start + batch_size, int(array.shape[0]))))
        loaded = [
            normalize_spatialdino_frame(array[frame, :, ::4, ::4].astype(np.float32))
            for frame in indices
        ]
        images = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
        probabilities = {
            name: predict_probability_batch(model, images, yx_tta=True).cpu().numpy()
            for name, model in models.items()
        }
        for batch_index, frame in enumerate(indices):
            combined = combine_probabilities(
                probabilities["feature24"][batch_index, 0],
                probabilities["feature36"][batch_index, 0],
                candidate,
            )
            rows.append(
                detections_from_probability(combined, frame=frame, candidate=candidate)
            )
        del images, probabilities
    return build_detection_cache(
        rows,
        candidate=candidate,
        estimated_node_count=estimated_count_for_movie(sample_path),
        calibration_frames=calibration_frames,
    )


@contextmanager
def substituted_official_detector(
    predict_module: ModuleType,
    cache: ExternalDetectionCache,
    *,
    coordinate_mode: str = "rounded",
) -> Iterator[None]:
    """Temporarily replace only the official baseline's detector callback."""

    original = getattr(predict_module, "_detect_cells_pooled", None)
    if not callable(original):
        raise TypeError("official predict module has no callable _detect_cells_pooled")

    def external_detector(
        _det_logits: Any,
        frame: int,
        _det_threshold: float,
        _pool_kernel: Sequence[int],
    ) -> np.ndarray:
        return cache.association_coords(int(frame), mode=coordinate_mode)

    predict_module._detect_cells_pooled = external_detector
    try:
        yield
    finally:
        predict_module._detect_cells_pooled = original


@contextmanager
def rounded_feature_sampling(models: Sequence[Any]) -> Iterator[None]:
    """Round subvoxel coordinates only at frozen feature-map lookups.

    The official linker was trained with integer-indexed node features. This
    adapter retains that contract while allowing its continuous positional and
    pairwise-coordinate branches to consume detector offsets.
    """

    records: list[tuple[Any, bool, Any]] = []
    try:
        for model in models:
            if model is None:
                continue
            original = getattr(model, "_index_features", None)
            if not callable(original):
                raise TypeError("association model has no callable _index_features")
            had_instance_attribute = "_index_features" in vars(model)
            previous_instance_value = vars(model).get("_index_features")

            def nearest_grid_features(
                feature_maps: Any,
                coordinates: Any,
                mask: Any,
                *,
                _original: Any = original,
            ) -> Any:
                return _original(feature_maps, coordinates.round(), mask)

            setattr(model, "_index_features", nearest_grid_features)
            records.append((model, had_instance_attribute, previous_instance_value))
        yield
    finally:
        for model, had_instance_attribute, previous_instance_value in reversed(records):
            if had_instance_attribute:
                setattr(model, "_index_features", previous_instance_value)
            else:
                delattr(model, "_index_features")


def predict_video_with_external_detections(
    predict_module: ModuleType,
    model: Any,
    sample_path: Path,
    device: Any,
    cfg: Any,
    cache: ExternalDetectionCache,
    association_coordinate_mode: str = "rounded",
    **kwargs: Any,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Call the official linker, validating and restoring detector positions."""

    if association_coordinate_mode not in ASSOCIATION_COORDINATE_MODES:
        raise ValueError(
            f"unsupported association coordinate mode: {association_coordinate_mode}"
        )
    with ExitStack() as stack:
        if association_coordinate_mode == "subvoxel":
            stack.enter_context(
                rounded_feature_sampling(
                    (model, kwargs.get("secondary_model"))
                )
            )
        stack.enter_context(
            substituted_official_detector(
                predict_module,
                cache,
                coordinate_mode=association_coordinate_mode,
            )
        )
        linked_coords, edges = predict_module.predict_video(
            model,
            sample_path,
            device,
            cfg,
            **kwargs,
        )
    return cache.restore_precise_output(linked_coords), edges
