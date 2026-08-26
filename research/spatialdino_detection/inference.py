"""Batched full-resolution inference for the hybrid SpatialDINO detector."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

try:
    from pu_targets import extract_local_peaks
    from train_spatialdino_pu_detector import normalize_spatialdino_frame
except ModuleNotFoundError:
    from research.spotiflow_biohub.pu_targets import extract_local_peaks
    from research.spatialdino_detection.train_pu_detector import (
        normalize_spatialdino_frame,
    )


LOW_PROBABILITY_THRESHOLD = 0.02


def refine_peaks_soft_centroid(
    probabilities: np.ndarray,
    coords: np.ndarray,
    *,
    radius: int = 1,
) -> np.ndarray:
    """Refine integer maxima with a local probability-weighted centroid."""

    volume = np.asarray(probabilities, dtype=np.float32)
    points = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("probabilities must be a finite 3D volume")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if not len(points) or radius == 0:
        return points.copy()
    refined = np.empty_like(points)
    shape = np.asarray(volume.shape, dtype=np.int64)
    for index, point in enumerate(points):
        center = np.rint(point).astype(np.int64)
        starts = np.maximum(center - radius, 0)
        stops = np.minimum(center + radius + 1, shape)
        axes = [np.arange(starts[axis], stops[axis], dtype=np.float32) for axis in range(3)]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        window = volume[tuple(slice(int(starts[a]), int(stops[a])) for a in range(3))]
        # Squared weights concentrate the estimate around a learned Gaussian
        # peak while still recovering its sub-voxel center.
        weights = np.square(np.clip(window, 0.0, None), dtype=np.float32)
        total = float(weights.sum())
        if total <= 1e-8:
            refined[index] = point
        else:
            refined[index] = np.asarray(
                [
                    float((zz * weights).sum()) / total,
                    float((yy * weights).sum()) / total,
                    float((xx * weights).sum()) / total,
                ],
                dtype=np.float32,
            )
    return refined


@torch.inference_mode()
def predict_probability_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    yx_tta: bool = True,
) -> torch.Tensor:
    if images.ndim != 5 or images.shape[1] != 1:
        raise ValueError("images must have shape (B, 1, Z, Y, X)")
    device_type = images.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=device_type == "cuda",
    ):
        logits = model(images)
        if yx_tta:
            for dims in ((-1,), (-2,), (-2, -1)):
                logits = logits + model(images.flip(dims)).flip(dims)
            logits = logits / 4.0
    return torch.sigmoid(logits.float())


def peaks_from_probability(
    probability: np.ndarray,
    *,
    threshold: float = LOW_PROBABILITY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    peak_set = extract_local_peaks(
        probability,
        threshold=threshold,
        min_distance_voxels=1,
    )
    refined = refine_peaks_soft_centroid(probability, peak_set.coords, radius=1)
    return refined, peak_set.confidence


def predict_frames(
    model: torch.nn.Module,
    sample_path: Path,
    frames: Iterable[int],
    *,
    device: torch.device,
    batch_size: int = 4,
    yx_tta: bool = True,
):
    """Return ``FramePeaks`` compatible rows without depending on Spotiflow."""

    try:
        from evaluate_pretrained_detector import FramePeaks
    except ModuleNotFoundError:
        from research.spotiflow_biohub.evaluate_pretrained_detector import FramePeaks

    import zarr

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    array = zarr.open_group(str(sample_path), mode="r")["0"]
    frame_indices = [int(frame) for frame in frames]
    results = []
    for start in range(0, len(frame_indices), batch_size):
        batch_frames = frame_indices[start : start + batch_size]
        loaded = [
            normalize_spatialdino_frame(array[frame, :, ::4, ::4].astype(np.float32))
            for frame in batch_frames
        ]
        images = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
        probabilities = predict_probability_batch(model, images, yx_tta=yx_tta).cpu().numpy()
        for batch_index, frame in enumerate(batch_frames):
            points, scores = peaks_from_probability(probabilities[batch_index, 0])
            results.append(FramePeaks(frame=frame, points_input=points, probabilities=scores))
        del images, probabilities
    return results, int(array.shape[0])
