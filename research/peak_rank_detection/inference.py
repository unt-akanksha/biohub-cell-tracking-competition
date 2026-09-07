"""D4 temporal inference for the independent peak-ranking detector."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


LOW_PROBABILITY_THRESHOLD = 0.01
TTA_TRANSFORMS = {
    "none": ((0, False),),
    "rot4": tuple((rotation, False) for rotation in range(4)),
    "d4": tuple(
        (rotation, flip_x)
        for flip_x in (False, True)
        for rotation in range(4)
    ),
}


def resolve_tta_mode(
    *, tta_mode: str | None = None, d4_tta: bool | None = None
) -> str:
    """Resolve the new explicit TTA contract while retaining old call sites."""

    if tta_mode is None:
        return "d4" if d4_tta is not False else "none"
    if tta_mode not in TTA_TRANSFORMS:
        raise ValueError(f"unsupported peak TTA mode: {tta_mode}")
    if d4_tta is not None and (tta_mode == "d4") != d4_tta:
        raise ValueError("d4_tta and tta_mode disagree")
    return tta_mode


def normalize_triplet(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError("frames must be a finite (3, Z, Y, X) array")
    low, high = np.quantile(values, (0.001, 0.999))
    if high <= low:
        raise ValueError("frame triplet has no intensity range")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _transform_frames(frames: torch.Tensor, *, rotation: int, flip_x: bool) -> torch.Tensor:
    values = frames.flip((-1,)) if flip_x else frames
    return torch.rot90(values, rotation % 4, dims=(-2, -1))


def _invert_scalar_field(
    values: torch.Tensor, *, rotation: int, flip_x: bool
) -> torch.Tensor:
    result = torch.rot90(values, -(rotation % 4), dims=(-2, -1))
    return result.flip((-1,)) if flip_x else result


def _invert_offset_field(
    values: torch.Tensor, *, rotation: int, flip_x: bool
) -> torch.Tensor:
    result = _invert_scalar_field(values, rotation=rotation, flip_x=flip_x)
    z, y, x = result[:, 0], result[:, 1], result[:, 2]
    for _ in range(rotation % 4):
        y, x = x, -y
    if flip_x:
        x = -x
    return torch.stack((z, y, x), dim=1)


@torch.inference_mode()
def predict_probability_and_offsets(
    model: torch.nn.Module,
    frames: torch.Tensor,
    *,
    d4_tta: bool | None = None,
    tta_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if frames.ndim != 5 or frames.shape[1] != 3:
        raise ValueError("frames must have shape (B, 3, Z, Y, X)")
    mode = resolve_tta_mode(tta_mode=tta_mode, d4_tta=d4_tta)
    transforms = TTA_TRANSFORMS[mode]
    logits_sum = None
    offset_sum = None
    device_type = frames.device.type
    for rotation, flip_x in transforms:
        augmented = _transform_frames(frames, rotation=rotation, flip_x=flip_x)
        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            output = model(augmented)
        logits = _invert_scalar_field(
            output["logits"].float(), rotation=rotation, flip_x=flip_x
        )
        offsets = _invert_offset_field(
            output["offsets"].float(), rotation=rotation, flip_x=flip_x
        )
        logits_sum = logits if logits_sum is None else logits_sum + logits
        offset_sum = offsets if offset_sum is None else offset_sum + offsets
    return torch.sigmoid(logits_sum / len(transforms)), offset_sum / len(transforms)


def peaks_from_prediction(
    probability: torch.Tensor,
    offsets: torch.Tensor,
    *,
    threshold: float = LOW_PROBABILITY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    if probability.ndim != 3 or offsets.shape != (3, *probability.shape):
        raise ValueError("probability and offset shapes differ")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in (0, 1)")
    maxima = probability == F.max_pool3d(
        probability[None, None], kernel_size=3, stride=1, padding=1
    )[0, 0]
    keep = maxima & (probability >= threshold)
    coords = torch.nonzero(keep, as_tuple=False)
    scores = probability[keep]
    if not len(coords):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    learned = offsets[:, coords[:, 0], coords[:, 1], coords[:, 2]].transpose(0, 1)
    refined = coords.float() + learned.clamp(-0.5, 0.5)
    return refined.cpu().numpy(), scores.cpu().numpy()


@torch.inference_mode()
def predict_frames(
    model: torch.nn.Module,
    sample_path: Path,
    frames: Iterable[int],
    *,
    device: torch.device,
    batch_size: int = 1,
    d4_tta: bool | None = None,
    tta_mode: str | None = None,
):
    try:
        from evaluate_pretrained_detector import FramePeaks
    except ModuleNotFoundError:
        from research.spotiflow_biohub.evaluate_pretrained_detector import FramePeaks

    import zarr

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    array = zarr.open_group(str(sample_path), mode="r")["0"]
    frame_count = int(array.shape[0])
    requested = [int(frame) for frame in frames]
    cache: dict[int, np.ndarray] = {}

    def pooled(frame: int) -> np.ndarray:
        frame = min(max(frame, 0), frame_count - 1)
        if frame not in cache:
            cache[frame] = array[frame, :, ::4, ::4].astype(np.float32)
            if cache[frame].shape != (64, 64, 64):
                raise ValueError(f"unexpected pooled frame shape: {cache[frame].shape}")
        return cache[frame]

    results = []
    for start in range(0, len(requested), batch_size):
        batch_frames = requested[start : start + batch_size]
        triplets = [
            normalize_triplet(
                np.stack((pooled(frame - 1), pooled(frame), pooled(frame + 1)))
            )
            for frame in batch_frames
        ]
        values = torch.from_numpy(np.stack(triplets)).to(device)
        probabilities, offsets = predict_probability_and_offsets(
            model, values, d4_tta=d4_tta, tta_mode=tta_mode
        )
        for index, frame in enumerate(batch_frames):
            points, scores = peaks_from_prediction(
                probabilities[index, 0], offsets[index]
            )
            results.append(
                FramePeaks(frame=frame, points_input=points, probabilities=scores)
            )
        # Sequential evaluation needs at most the local temporal neighborhood.
        minimum_needed = min(batch_frames) - 1
        cache = {key: value for key, value in cache.items() if key >= minimum_needed}
    return results, frame_count
