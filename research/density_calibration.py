"""Conservative, image-backed detector-density calibration for Biohub.

The competition inputs expose an ``estimated_number_of_nodes`` field.  The
public inference pipeline currently uses that value only in its local scorer,
while applying one fixed detection threshold to every movie.  This module uses
uniformly sampled detector responses to raise that threshold only when the
movie is projected to be over-detected.  It intentionally does not lower the
base threshold: under-counted movies keep the clean baseline behavior until a
detector can demonstrate extra recall on held-out ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DensityCalibration:
    threshold: float
    base_projected_count: float
    calibrated_projected_count: float
    target_raw_count: float
    sampled_frames: int
    reason: str


def _find_key(value: Any, key: str) -> Any | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def read_estimated_node_count(path: Path) -> float | None:
    """Read the organizer-provided count estimate from Zarr or GEFF metadata."""

    for candidate in (path / "zarr.json", path / ".zattrs"):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        found = _find_key(payload, "estimated_number_of_nodes")
        if found is None:
            continue
        try:
            result = float(found)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result) and result > 0:
            return result
    return None


def uniform_frame_indices(n_frames: int, n_samples: int = 12) -> np.ndarray:
    """Choose deterministic, approximately midpoint-stratified movie frames."""

    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    n_samples = min(n_frames, n_samples)
    edges = np.linspace(0.0, float(n_frames), n_samples + 1)
    indices = np.floor((edges[:-1] + edges[1:]) / 2.0).astype(np.int64)
    return np.clip(indices, 0, n_frames - 1)


def _project_count(
    peak_probabilities: Sequence[np.ndarray], threshold: float, n_frames: int
) -> float:
    sampled = len(peak_probabilities)
    if sampled == 0:
        raise ValueError("at least one sampled frame is required")
    selected = sum(
        int(np.count_nonzero(np.asarray(values, dtype=np.float64) > threshold))
        for values in peak_probabilities
    )
    return float(selected) * float(n_frames) / float(sampled)


def select_conservative_threshold(
    peak_probabilities: Sequence[np.ndarray],
    *,
    estimated_node_count: float,
    n_frames: int,
    base_threshold: float,
    postprocess_retention: float = 0.96,
    overcount_tolerance: float = 0.03,
    max_threshold_increase: float = 0.025,
) -> DensityCalibration:
    """Select an over-detection-only threshold from sampled local maxima.

    ``peak_probabilities`` contains the detector probabilities at local maxima
    for uniformly sampled frames.  ``postprocess_retention`` converts the
    desired final graph size into the corresponding raw detector count.
    """

    if not 0.0 < base_threshold < 1.0:
        raise ValueError("base_threshold must be between zero and one")
    if not np.isfinite(estimated_node_count) or estimated_node_count <= 0:
        raise ValueError("estimated_node_count must be finite and positive")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not 0.0 < postprocess_retention <= 1.0:
        raise ValueError("postprocess_retention must be in (0, 1]")
    if overcount_tolerance < 0.0:
        raise ValueError("overcount_tolerance must be non-negative")
    if max_threshold_increase < 0.0:
        raise ValueError("max_threshold_increase must be non-negative")

    arrays = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in peak_probabilities
    ]
    if not arrays:
        raise ValueError("at least one sampled frame is required")
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("peak probabilities must be finite")

    target_raw = float(estimated_node_count) / postprocess_retention
    base_projected = _project_count(arrays, base_threshold, n_frames)
    if base_projected <= target_raw * (1.0 + overcount_tolerance):
        return DensityCalibration(
            threshold=float(base_threshold),
            base_projected_count=base_projected,
            calibrated_projected_count=base_projected,
            target_raw_count=target_raw,
            sampled_frames=len(arrays),
            reason="base_not_overcounted",
        )

    eligible = np.concatenate([values[values > base_threshold] for values in arrays])
    desired_sample_count = int(round(target_raw * len(arrays) / n_frames))
    desired_sample_count = max(1, min(desired_sample_count, len(eligible)))
    # A strict ``probability > threshold`` comparison retains exactly the top
    # desired values when scores are distinct.  Ties are harmless and reported
    # through the projected count rather than hidden by random tie-breaking.
    ordered = np.sort(eligible)[::-1]
    boundary = float(ordered[desired_sample_count - 1])
    if desired_sample_count < len(ordered):
        next_value = float(ordered[desired_sample_count])
        candidate = (boundary + next_value) / 2.0
    else:
        candidate = float(base_threshold)

    cap = min(1.0 - np.finfo(np.float64).eps, base_threshold + max_threshold_increase)
    threshold = float(np.clip(candidate, base_threshold, cap))
    projected = _project_count(arrays, threshold, n_frames)
    reason = "raised_to_density_target" if threshold < cap else "raised_to_safety_cap"
    return DensityCalibration(
        threshold=threshold,
        base_projected_count=base_projected,
        calibrated_projected_count=projected,
        target_raw_count=target_raw,
        sampled_frames=len(arrays),
        reason=reason,
    )


def flatten_peak_probabilities(values: Iterable[Iterable[float]]) -> list[np.ndarray]:
    """Normalize notebook-produced probability lists for the selector."""

    return [np.asarray(frame, dtype=np.float64).reshape(-1) for frame in values]
