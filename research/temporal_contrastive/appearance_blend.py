"""Inference utilities for blending learned 3D appearance with Trackastra."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from patch_model import (
        PhysicalPatchAssociationModel,
        sample_physical_patches,
        temporal_context_volume,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.patch_model import (
        PhysicalPatchAssociationModel,
        sample_physical_patches,
        temporal_context_volume,
    )


def blend_pair_scores(
    trackastra_scores: np.ndarray,
    appearance_cosine: np.ndarray,
    *,
    appearance_weight: float,
    appearance_temperature: float = 0.10,
    source_division_logits: np.ndarray | None = None,
    division_weight: float = 0.0,
) -> np.ndarray:
    """Add appearance and source-division evidence in log-odds space.

    Non-finite Trackastra entries remain unavailable. A zero appearance weight
    and zero division weight return an exact copy of the Trackastra matrix,
    making the unmodified model an explicit calibration candidate. Appearance
    evidence is row-centered because it ranks candidate children. Division
    evidence is constant within a source row because it changes the confidence
    that a second child should clear the frozen association thresholds.
    """

    track = np.asarray(trackastra_scores, dtype=np.float32)
    appearance = np.asarray(appearance_cosine, dtype=np.float32)
    if track.shape != appearance.shape or track.ndim != 2:
        raise ValueError("Trackastra and appearance scores must be equal matrices")
    if not np.isfinite(appearance).all():
        raise ValueError("appearance scores must be finite")
    if appearance_weight < 0:
        raise ValueError("appearance_weight cannot be negative")
    if division_weight < 0:
        raise ValueError("division_weight cannot be negative")
    if appearance_temperature <= 0:
        raise ValueError("appearance_temperature must be positive")
    division = None
    if source_division_logits is not None:
        division = np.asarray(source_division_logits, dtype=np.float32)
        if division.shape != (track.shape[0],):
            raise ValueError("one source division logit is required per score row")
        if not np.isfinite(division).all():
            raise ValueError("source division logits must be finite")
    if division_weight > 0 and division is None:
        raise ValueError("positive division_weight requires source division logits")
    available = np.isfinite(track)
    if np.any(available & ((track < 0) | (track > 1))):
        raise ValueError("finite Trackastra probabilities must be in [0, 1]")
    if appearance_weight == 0 and division_weight == 0:
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
    if division_weight > 0:
        division_evidence = np.clip(division, -8.0, 8.0)[:, None]
        combined = combined + float(division_weight) * division_evidence
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
    frame_cache: dict[int, np.ndarray | torch.Tensor] = {}
    model.eval()
    for timepoint in sorted(int(value) for value in np.unique(video.times)):
        rows = np.flatnonzero(video.times == timepoint)
        if timepoint < 0 or timepoint >= int(image_array.shape[0]):
            raise ValueError(f"node frame is outside image array: {timepoint}")
        frame = torch.as_tensor(
            np.asarray(
                temporal_context_volume(
                    image_array, timepoint, frame_cache=frame_cache
                )
            ),
            device=device,
        )
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
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint:
                del frame_cache[cached_timepoint]
    if not np.isfinite(embeddings).all() or not np.isfinite(division_logits).all():
        raise RuntimeError("appearance model produced non-finite node outputs")
    return embeddings, division_logits, {
        "nodes": len(video.node_ids),
        "frames": len(frame_rows),
        "nodes_by_frame": frame_rows,
        "embedding_width": embedding_width,
    }


@torch.inference_mode()
def extract_reciprocal_movie_embeddings(
    primary_model: PhysicalPatchAssociationModel,
    peer_model: PhysicalPatchAssociationModel,
    video: Any,
    image_array: Any,
    device: torch.device,
    *,
    voxel_size_zyx_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    node_batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Encode two reciprocal models while sampling every physical patch once."""

    if node_batch_size <= 0:
        raise ValueError("node_batch_size must be positive")
    if len(video.node_ids) != len(video.times) or len(video.node_ids) != len(
        video.coords_voxel
    ):
        raise ValueError("video node arrays are misaligned")
    widths = (
        int(primary_model.projection[-1].out_features),
        int(peer_model.projection[-1].out_features),
    )
    if widths[0] != widths[1]:
        raise ValueError("reciprocal appearance embedding widths differ")
    primary_embeddings = np.empty((len(video.node_ids), widths[0]), dtype=np.float32)
    peer_embeddings = np.empty_like(primary_embeddings)
    primary_divisions = np.empty(len(video.node_ids), dtype=np.float32)
    peer_divisions = np.empty_like(primary_divisions)
    frame_rows: dict[int, int] = {}
    frame_cache: dict[int, np.ndarray | torch.Tensor] = {}
    primary_model.eval()
    peer_model.eval()
    for timepoint in sorted(int(value) for value in np.unique(video.times)):
        rows = np.flatnonzero(video.times == timepoint)
        if timepoint < 0 or timepoint >= int(image_array.shape[0]):
            raise ValueError(f"node frame is outside image array: {timepoint}")
        frame = torch.as_tensor(
            np.asarray(
                temporal_context_volume(
                    image_array, timepoint, frame_cache=frame_cache
                )
            ),
            device=device,
        )
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
                primary_encoded, primary_logits = primary_model(patches)
                peer_encoded, peer_logits = peer_model(patches)
            primary_embeddings[selected_rows] = primary_encoded.float().cpu().numpy()
            peer_embeddings[selected_rows] = peer_encoded.float().cpu().numpy()
            primary_divisions[selected_rows] = primary_logits.float().cpu().numpy()
            peer_divisions[selected_rows] = peer_logits.float().cpu().numpy()
        frame_rows[timepoint] = len(rows)
        for cached_timepoint in list(frame_cache):
            if cached_timepoint < timepoint:
                del frame_cache[cached_timepoint]
    outputs = (
        primary_embeddings,
        primary_divisions,
        peer_embeddings,
        peer_divisions,
    )
    if any(not np.isfinite(output).all() for output in outputs):
        raise RuntimeError("reciprocal appearance model produced non-finite outputs")
    return (*outputs, {
        "nodes": len(video.node_ids),
        "frames": len(frame_rows),
        "nodes_by_frame": frame_rows,
        "embedding_width": widths[0],
        "encoder_count": 2,
        "physical_patch_extractions_per_node": 1,
    })


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


def division_logits_for_movie(
    video: Any,
    node_division_logits: np.ndarray,
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Align one learned division logit to each Trackastra source row."""

    divisions = np.asarray(node_division_logits, dtype=np.float32)
    if divisions.shape != (len(video.node_ids),):
        raise ValueError("node division logits do not align with the video")
    if not np.isfinite(divisions).all():
        raise ValueError("node division logits must be finite")
    id_to_row = {
        int(node_id): row for row, node_id in enumerate(video.node_ids.tolist())
    }
    result: dict[int, np.ndarray] = {}
    for timepoint, (source_ids, _target_ids, track_scores) in pair_scores.items():
        source_rows = np.asarray([id_to_row[int(node)] for node in source_ids])
        aligned = divisions[source_rows]
        if aligned.shape != (track_scores.shape[0],):
            raise RuntimeError("division logits and Trackastra source rows diverged")
        result[int(timepoint)] = aligned.astype(np.float32, copy=False)
    return result


def reciprocal_movie_evidence(
    primary_appearance: dict[int, np.ndarray],
    primary_divisions: dict[int, np.ndarray],
    peer_appearance: dict[int, np.ndarray],
    peer_divisions: dict[int, np.ndarray],
    *,
    mode: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Select one reciprocal model or an equal-probability clean ensemble."""

    expected = set(primary_appearance)
    if not (
        expected == set(primary_divisions)
        and expected == set(peer_appearance)
        and expected == set(peer_divisions)
    ):
        raise ValueError("reciprocal appearance evidence has different transitions")
    if mode not in {"target_only", "reciprocal_mean"}:
        raise ValueError(f"unknown reciprocal appearance mode: {mode}")
    appearance: dict[int, np.ndarray] = {}
    divisions: dict[int, np.ndarray] = {}
    for timepoint in expected:
        primary_scores = np.asarray(primary_appearance[timepoint], dtype=np.float32)
        primary_logits = np.asarray(primary_divisions[timepoint], dtype=np.float32)
        peer_scores = np.asarray(peer_appearance[timepoint], dtype=np.float32)
        peer_logits = np.asarray(peer_divisions[timepoint], dtype=np.float32)
        if primary_scores.shape != peer_scores.shape:
            raise ValueError("reciprocal appearance matrices have different shapes")
        if primary_logits.shape != peer_logits.shape:
            raise ValueError("reciprocal division vectors have different shapes")
        if mode == "target_only":
            appearance[timepoint] = primary_scores.copy()
            divisions[timepoint] = primary_logits.copy()
        else:
            appearance[timepoint] = (
                0.5 * primary_scores + 0.5 * peer_scores
            ).astype(np.float32)
            divisions[timepoint] = (
                0.5 * primary_logits + 0.5 * peer_logits
            ).astype(np.float32)
    return appearance, divisions


def blend_movie_pair_scores(
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    appearance_scores: dict[int, np.ndarray],
    *,
    appearance_weight: float,
    appearance_temperature: float = 0.10,
    source_division_logits: dict[int, np.ndarray] | None = None,
    division_weight: float = 0.0,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if set(pair_scores) != set(appearance_scores):
        raise ValueError("appearance scores do not cover every frame transition")
    if source_division_logits is not None and set(pair_scores) != set(
        source_division_logits
    ):
        raise ValueError("division logits do not cover every frame transition")
    if division_weight > 0 and source_division_logits is None:
        raise ValueError("positive division_weight requires movie division logits")
    return {
        timepoint: (
            source_ids,
            target_ids,
            blend_pair_scores(
                scores,
                appearance_scores[timepoint],
                appearance_weight=appearance_weight,
                appearance_temperature=appearance_temperature,
                source_division_logits=(
                    None
                    if source_division_logits is None
                    else source_division_logits[timepoint]
                ),
                division_weight=division_weight,
            ),
        )
        for timepoint, (source_ids, target_ids, scores) in pair_scores.items()
    }
