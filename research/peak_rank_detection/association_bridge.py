"""Route peak-ranking detections through the frozen official association model.

The detector owns node locations and confidence.  The official linker owns only
edge inference; after linking, sub-voxel detector coordinates are restored.
Density calibration uses organizer metadata and never reads labels or test
answers.
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
        build_detection_cache,
        estimated_count_for_movie,
        predict_video_with_external_detections,
    )
except ModuleNotFoundError:
    from lsm_association_bridge import (
        DetectorCandidate,
        ExternalDetectionCache,
        FrameDetections,
        build_detection_cache,
        estimated_count_for_movie,
        predict_video_with_external_detections,
    )


PEAK_RANK_CANDIDATE = DetectorCandidate(
    name="temporal_peak_rank_v1",
    feature24_weight=1.0,
    feature36_weight=0.0,
    refinement="centroid",
)


def predict_movie_detection_cache(
    model: Any,
    sample_path: Path,
    *,
    device: Any,
    batch_size: int = 1,
    calibration_frames: int = 12,
    d4_tta: bool | None = None,
    tta_mode: str | None = None,
) -> ExternalDetectionCache:
    """Infer every frame and freeze one metadata-calibrated movie threshold."""

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
    return build_detection_cache(
        frames,
        candidate=PEAK_RANK_CANDIDATE,
        estimated_node_count=estimated_count_for_movie(sample_path),
        calibration_frames=calibration_frames,
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
    batch_size: int = 1,
    calibration_frames: int = 12,
    d4_tta: bool | None = None,
    tta_mode: str | None = None,
    **association_kwargs: Any,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Compose the independent detector with unchanged official edge inference."""

    cache = predict_movie_detection_cache(
        detector_model,
        sample_path,
        device=device if detector_device is None else detector_device,
        batch_size=batch_size,
        calibration_frames=calibration_frames,
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
