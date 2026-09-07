"""Route peak-ranking detections through the frozen official association model.

The detector owns node locations and confidence.  The official linker owns only
edge inference; after linking, sub-voxel detector coordinates are restored.
One checkpoint-bound global threshold is frozen on complete synthetic selection
labels before competition validation.  Organizer node-count metadata is retained
only as a diagnostic and never changes predicted nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from research.peak_rank_detection.inference import predict_frames
except ModuleNotFoundError:
    from inference import predict_frames

try:
    from research.lsm_fm_detection.association_bridge import (
        DetectorCandidate,
        ExternalDetectionCache,
        FrameDetections,
        predict_video_with_external_detections,
    )
except ModuleNotFoundError:
    from lsm_association_bridge import (
        DetectorCandidate,
        ExternalDetectionCache,
        FrameDetections,
        predict_video_with_external_detections,
    )


PEAK_RANK_CANDIDATE = DetectorCandidate(
    name="temporal_peak_rank_v1",
    feature24_weight=1.0,
    feature36_weight=0.0,
    refinement="centroid",
)
PEAK_THRESHOLD_POLICY = "synthetic_selection_micro_detection_jaccard"


def predict_movie_detection_cache(
    model: Any,
    sample_path: Path,
    *,
    device: Any,
    peak_threshold: float,
    batch_size: int = 1,
    d4_tta: bool | None = None,
    tta_mode: str | None = None,
) -> ExternalDetectionCache:
    """Infer every frame using one clean, globally frozen score threshold."""

    if not np.isfinite(peak_threshold) or not 0.0 < peak_threshold < 1.0:
        raise ValueError("peak threshold must be finite and lie in (0, 1)")

    predictions, frame_count = predict_frames(
        model,
        sample_path,
        range_frame_count(sample_path),
        device=device,
        batch_size=batch_size,
        d4_tta=d4_tta,
        tta_mode=tta_mode,
    )
    if int(frame_count) != len(predictions):
        raise RuntimeError("peak-ranking inference did not cover every movie frame")
    frames = tuple(
        FrameDetections(
            frame=int(row.frame),
            points_input=np.asarray(row.points_input, dtype=np.float32),
            confidence=np.asarray(row.probabilities, dtype=np.float32),
        )
        for row in predictions
    )
    if tuple(row.frame for row in frames) != tuple(range(int(frame_count))):
        raise RuntimeError("peak-ranking frame ordering changed")
    selected_count = sum(
        int(np.count_nonzero(row.confidence > peak_threshold)) for row in frames
    )
    return ExternalDetectionCache(
        candidate=PEAK_RANK_CANDIDATE,
        frames=frames,
        threshold=float(peak_threshold),
        projected_node_count=float(selected_count),
        # Prediction generation intentionally never reads the organizer's
        # estimated node count.  The patched scorer may consume it separately.
        estimated_node_count=None,
    )


def range_frame_count(sample_path: Path) -> range:
    """Read only image shape so ``predict_frames`` receives the full movie."""

    import zarr

    array = zarr.open_group(str(sample_path), mode="r")["0"]
    return range(int(array.shape[0]))


def predict_video_with_peak_rank_detections(
    predict_module: Any,
    association_model: Any,
    detector_model: Any,
    sample_path: Path,
    device: Any,
    cfg: Any,
    *,
    detector_device: Any | None = None,
    peak_threshold: float,
    batch_size: int = 1,
    d4_tta: bool | None = None,
    tta_mode: str | None = None,
    **association_kwargs: Any,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Compose the independent detector with unchanged official edge inference."""

    cache = predict_movie_detection_cache(
        detector_model,
        sample_path,
        device=device if detector_device is None else detector_device,
        peak_threshold=peak_threshold,
        batch_size=batch_size,
        d4_tta=d4_tta,
        tta_mode=tta_mode,
    )
    return predict_video_with_external_detections(
        predict_module,
        association_model,
        sample_path,
        device,
        cfg,
        cache,
        **association_kwargs,
    )
