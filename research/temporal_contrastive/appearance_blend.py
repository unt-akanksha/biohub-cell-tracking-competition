"""Inference utilities for blending learned 3D appearance with Trackastra."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from patch_model import PhysicalPatchAssociationModel, sample_physical_patches
except ModuleNotFoundError:
    from research.temporal_contrastive.patch_model import (
        PhysicalPatchAssociationModel,
        sample_physical_patches,
    )


def blend_pair_scores(
    trackastra_scores: np.ndarray,
    appearance_cosine: np.ndarray,
    *,
    appearance_weight: float,
    appearance_temperature: float = 0.10,
) -> np.ndarray:
    """Add row-centered appearance evidence in log-odds space.

    Non-finite Trackastra entries remain unavailable. A zero appearance weight
    returns an exact copy of the Trackastra matrix, making the unmodified model
    an explicit calibration candidate.
    """

    track = np.asarray(trackastra_scores, dtype=np.float32)
    appearance = np.asarray(appearance_cosine, dtype=np.float32)
    if track.shape != appearance.shape or track.ndim != 2:
        raise ValueError("Trackastra and appearance scores must be equal matrices")
    if not np.isfinite(appearance).all():
        raise ValueError("appearance scores must be finite")
    if appearance_weight < 0:
        raise ValueError("appearance_weight cannot be negative")
    if appearance_temperature <= 0:
        raise ValueError("appearance_temperature must be positive")
    available = np.isfinite(track)
    if np.any(available & ((track < 0) | (track > 1))):
        raise ValueError("finite Trackastra probabilities must be in [0, 1]")
    if appearance_weight == 0:
        return track.copy()

    centered = np.zeros_like(appearance)
    for row in range(len(appearance)):
        candidates = available[row]
        if not np.any(candidates):
            continue
        values = appearance[row, candidates]
        centered[row, candidates] = (
            values - float(values.mean())
        ) / float(appearance_temperature)
    centered = np.clip(centered, -8.0, 8.0)
    clipped = np.clip(track, 1e-5, 1.0 - 1e-5)
    track_logits = np.log(clipped) - np.log1p(-clipped)
    combined = track_logits + float(appearance_weight) * centered
    output = (1.0 / (1.0 + np.exp(-combined))).astype(np.float32)
    output[~available] = -np.inf
    return output


@torch.inference_mode()
def extract_movie_embeddings(
    model: PhysicalPatchAssociationModel,
    video: Any,
    image_array: Any,
    device: torch.device,
    *,
    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    node_batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Encode every submitted node once, grouped by its complete movie frame."""

    if node_batch_size <= 0:
        raise ValueError("node_batch_size must be positive")
    if len(video.node_ids) != len(video.times) or len(video.node_ids) != len(video.coords_voxel):
        raise ValueError("video node arrays are misaligned")
    embedding_width = int(model.projection[-1].out_features)
    embeddings = np.empty((len(video.node_ids), embedding_width), dtype=np.float32)
    division_logits = np.empty(len(video.node_ids), dtype=np.float32)
    frame_rows: dict[int, int] = {}
    model.eval()
    for timepoint in sorted(int(value) for value in np.unique(video.times)):
        rows = np.flatnonzero(video.times == timepoint)
        if timepoint < 0 or timepoint >= int(image_array.shape[0]):
            raise ValueError(f"node frame is outside image array: {timepoint}")
        frame = torch.as_tensor(np.asarray(image_array[timepoint]), device=device)
        for start in range(0, len(rows), node_batch_size):
            selected_rows = rows[start : start + node_batch_size]
            patches = sample_physical_patches(
                frame,
                video.coords_voxel[selected_rows],
                voxel_size_zyx_um=voxel_size_zyx_um,
                chunk_size=node_batch_size,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                encoded, divisions = model(patches)
            embeddings[selected_rows] = encoded.float().cpu().numpy()
            division_logits[selected_rows] = divisions.float().cpu().numpy()
        frame_rows[timepoint] = len(rows)
    if not np.isfinite(embeddings).all() or not np.isfinite(division_logits).all():
        raise RuntimeError("appearance model produced non-finite node outputs")
    return embeddings, division_logits, {
        "nodes": len(video.node_ids),
        "frames": len(frame_rows),
        "nodes_by_frame": frame_rows,
        "embedding_width": embedding_width,
    }


def appearance_scores_for_movie(
    video: Any,
    node_embeddings: np.ndarray,
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[int, np.ndarray]:
    embeddings = np.asarray(node_embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(video.node_ids):
        raise ValueError("node embeddings do not align with the video")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    id_to_row = {
        int(node_id): row for row, node_id in enumerate(video.node_ids.tolist())
    }
    result: dict[int, np.ndarray] = {}
    for timepoint, (source_ids, target_ids, track_scores) in pair_scores.items():
        source_rows = np.asarray([id_to_row[int(node)] for node in source_ids])
        target_rows = np.asarray([id_to_row[int(node)] for node in target_ids])
        appearance = embeddings[source_rows] @ embeddings[target_rows].T
        if appearance.shape != track_scores.shape:
            raise RuntimeError("appearance and Trackastra pair inventories diverged")
        result[int(timepoint)] = np.clip(appearance, -1.0, 1.0).astype(np.float32)
    return result


def blend_movie_pair_scores(
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    appearance_scores: dict[int, np.ndarray],
    *,
    appearance_weight: float,
    appearance_temperature: float = 0.10,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if set(pair_scores) != set(appearance_scores):
        raise ValueError("appearance scores do not cover every frame transition")
    return {
        timepoint: (
            source_ids,
            target_ids,
            blend_pair_scores(
                scores,
                appearance_scores[timepoint],
                appearance_weight=appearance_weight,
                appearance_temperature=appearance_temperature,
            ),
        )
        for timepoint, (source_ids, target_ids, scores) in pair_scores.items()
    }
