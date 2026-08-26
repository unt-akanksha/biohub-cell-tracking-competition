"""Feature-diverse Biohub linker using official models and owned detections.

This module is a clean integration around the BSD-3-Clause organizer baseline
at commit ``075fc5f5a52d11077f9dc2b074644618f26939e2``.  It does not copy the
public 0.927 notebook's prediction program.  Instead, it reuses the official
model API and combines two CC0 checkpoints with a predeclared reciprocal
probability consensus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from research.lsm_fm_detection.association_bridge import (
    ExternalDetectionCache,
    SPATIAL_DOWNSAMPLE,
)


OFFICIAL_SOURCE_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL_SOURCE_LICENSE = "BSD-3-Clause"
PRIMARY_CHECKPOINT_SHA256 = (
    "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
)
SECONDARY_CHECKPOINT_SHA256 = (
    "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
)


@dataclass(frozen=True)
class AssociationConsensusConfig:
    """Frozen association consensus; no per-movie or leaderboard tuning."""

    edge_threshold: float = 0.48
    secondary_seed_weight: float = 0.15
    reverse_direction_weight: float = 0.30
    window_size: int = 2
    downsample: tuple[int, int, int] = (1, 4, 4)

    def __post_init__(self) -> None:
        if not 0.0 < self.edge_threshold < 1.0:
            raise ValueError("edge_threshold must lie in (0, 1)")
        if not 0.0 < self.secondary_seed_weight < 0.5:
            raise ValueError("secondary_seed_weight must lie in (0, 0.5)")
        if not 0.0 < self.reverse_direction_weight < 0.5:
            raise ValueError("reverse_direction_weight must lie in (0, 0.5)")
        if self.window_size < 2:
            raise ValueError("window_size must be at least two")
        if self.downsample != (1, 4, 4):
            raise ValueError("LSM-FM association requires downsample (1, 4, 4)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"association checkpoint hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )


def _weighted_harmonic(
    first: Any,
    second: Any,
    *,
    second_weight: float,
    normalize_dim: int,
) -> Any:
    if first.shape != second.shape:
        raise ValueError("consensus probability shapes differ")
    if not 0.0 < second_weight < 1.0:
        raise ValueError("second_weight must lie in (0, 1)")
    first_probability = first.float().clamp_min(1e-8)
    second_probability = second.float().clamp_min(1e-8)
    result = 1.0 / (
        (1.0 - second_weight) / first_probability
        + second_weight / second_probability
    )
    return result / result.sum(dim=normalize_dim, keepdim=True).clamp_min(1e-8)


def reciprocal_seed_consensus(
    primary_forward_logits: Any,
    primary_reverse_logits: Any,
    secondary_forward_logits: Any,
    secondary_reverse_logits: Any,
    *,
    secondary_seed_weight: float = 0.15,
    reverse_direction_weight: float = 0.30,
) -> Any:
    """Fuse parent-normalized and child-normalized evidence from two seeds.

    Forward logits have shape ``(B, N_source, N_target)``. Reverse logits are
    native reverse-model outputs with shape ``(B, N_target, N_source)``. The
    weighted harmonic mean is deliberately conservative: one seed or temporal
    direction cannot rescue an edge that another assigns negligible support.
    """

    expected = primary_forward_logits.shape
    if secondary_forward_logits.shape != expected:
        raise ValueError("forward seed logit shapes differ")
    reverse_expected = (expected[0], expected[2], expected[1])
    if (
        primary_reverse_logits.shape != reverse_expected
        or secondary_reverse_logits.shape != reverse_expected
    ):
        raise ValueError("reverse logits do not transpose onto forward edges")
    primary_forward = torch.softmax(primary_forward_logits.float(), dim=1)
    secondary_forward = torch.softmax(secondary_forward_logits.float(), dim=1)
    primary_reverse = torch.softmax(primary_reverse_logits.float(), dim=1).transpose(1, 2)
    secondary_reverse = torch.softmax(secondary_reverse_logits.float(), dim=1).transpose(1, 2)
    primary = _weighted_harmonic(
        primary_forward,
        primary_reverse,
        second_weight=reverse_direction_weight,
        normalize_dim=1,
    )
    secondary = _weighted_harmonic(
        secondary_forward,
        secondary_reverse,
        second_weight=reverse_direction_weight,
        normalize_dim=1,
    )
    return _weighted_harmonic(
        primary,
        secondary,
        second_weight=secondary_seed_weight,
        normalize_dim=1,
    )


def _encode_window(models: dict[str, Any], images: Any) -> dict[str, Any]:
    encoded = {}
    for name, model in models.items():
        features, detection_logits = model.encode(images)
        del detection_logits
        encoded[name] = features
    return encoded


def _edge_probabilities(
    models: dict[str, Any],
    encoded: dict[str, Any],
    *,
    frame_offset: int,
    coordinates_source: Any,
    coordinates_target: Any,
    positions_source: Any,
    positions_target: Any,
    mask_source: Any,
    mask_target: Any,
    downsample_tensor: Any,
    config: AssociationConsensusConfig,
) -> Any:
    logits: dict[str, tuple[Any, Any]] = {}
    for name, model in models.items():
        source_features = model._index_features(
            encoded[name][:, frame_offset], coordinates_source, mask_source
        )
        target_features = model._index_features(
            encoded[name][:, frame_offset + 1], coordinates_target, mask_target
        )
        forward = model.predict_edges(
            source_features,
            target_features,
            coordinates_source * downsample_tensor,
            coordinates_target * downsample_tensor,
            positions_source,
            positions_target,
            mask_source,
            mask_target,
        )
        reverse = model.predict_edges(
            target_features,
            source_features,
            coordinates_target * downsample_tensor,
            coordinates_source * downsample_tensor,
            positions_target,
            positions_source,
            mask_target,
            mask_source,
        )
        logits[name] = (forward, reverse)
    return reciprocal_seed_consensus(
        logits["primary"][0],
        logits["primary"][1],
        logits["secondary"][0],
        logits["secondary"][1],
        secondary_seed_weight=config.secondary_seed_weight,
        reverse_direction_weight=config.reverse_direction_weight,
    )


@torch.inference_mode()
def predict_video_consensus(
    official_predict: ModuleType,
    models: dict[str, Any],
    sample_path: Path,
    device: Any,
    cache: ExternalDetectionCache,
    *,
    config: AssociationConsensusConfig = AssociationConsensusConfig(),
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], dict[str, Any]]:
    """Predict a complete movie with external nodes and reciprocal association."""

    import zarr

    if set(models) != {"primary", "secondary"}:
        raise ValueError("models must contain exactly primary and secondary")
    if tuple(SPATIAL_DOWNSAMPLE.astype(int)) != config.downsample:
        raise RuntimeError("detector and linker downsample grids differ")
    dataset = official_predict.open_dataset(
        sample_path,
        normalize=False,
        load_image=False,
        downsample=config.downsample,
    )
    if "0.001" not in dataset.quantiles or "0.999" not in dataset.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {sample_path}")
    array = zarr.open_group(str(dataset.zarr_path), mode="r")["0"]
    frame_count = int(dataset.image_shape[0])
    if len(cache.frames) != frame_count:
        raise RuntimeError("external detections do not cover the complete movie")
    target_shape = list(dataset.image_shape[1:])
    q_low = float(dataset.quantiles["0.001"])
    q_high = float(dataset.quantiles["0.999"])
    downsample_array = np.asarray(config.downsample, dtype=np.float32)
    downsample_tensor = torch.from_numpy(downsample_array).to(device)
    image_shape = (frame_count,) + tuple(dataset.image_shape[1:])

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offsets: dict[int, tuple[int, int]] = {}
    node_count = 0
    edges: list[tuple[int, int, float, float]] = []
    window_size = config.window_size
    stride = max(window_size - 1, 1)
    window_starts = list(range(0, frame_count - window_size + 1, stride))
    if not window_starts or window_starts[-1] + window_size < frame_count:
        last = max(frame_count - window_size, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for window_start in window_starts:
        frame_indices = list(range(window_start, window_start + window_size))
        images = torch.stack(
            [
                official_predict._load_frame(
                    array, frame, target_shape, config.downsample
                )
                for frame in frame_indices
            ]
        )
        images = ((images - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        images = images.unsqueeze(0).to(device)
        encoded = _encode_window(models, images)
        del images

        for frame in frame_indices:
            if frame in seen_frames:
                continue
            coordinates = cache.association_coords(frame)
            coord_offsets[frame] = (node_count, node_count + len(coordinates))
            node_count += len(coordinates)
            coord_lists.append(coordinates)
            seen_frames.add(frame)

        coords_so_far = (
            np.concatenate(coord_lists)
            if coord_lists
            else np.empty((0, 4), dtype=np.int16)
        )
        for frame_offset in range(window_size - 1):
            source_frame = frame_indices[frame_offset]
            target_frame = frame_indices[frame_offset + 1]
            if (source_frame, target_frame) in seen_pairs:
                continue
            seen_pairs.add((source_frame, target_frame))
            source_start, source_stop = coord_offsets[source_frame]
            target_start, target_stop = coord_offsets[target_frame]
            if source_start == source_stop or target_start == target_stop:
                continue
            source = coords_so_far[source_start:source_stop]
            target = coords_so_far[target_start:target_stop]
            source_indices = np.arange(source_start, source_stop, dtype=np.int64)
            target_indices = np.arange(target_start, target_stop, dtype=np.int64)
            source_coordinates = torch.from_numpy(
                source[:, 1:].astype(np.float32)
            ).unsqueeze(0).to(device)
            target_coordinates = torch.from_numpy(
                target[:, 1:].astype(np.float32)
            ).unsqueeze(0).to(device)
            relative_source = source.copy()
            relative_target = target.copy()
            relative_source[:, 0] = frame_offset
            relative_target[:, 0] = frame_offset + 1
            relative_shape = (window_size,) + image_shape[1:]
            source_positions = torch.from_numpy(
                official_predict.extract_pos_features(relative_source, relative_shape)
            ).unsqueeze(0).to(device)
            target_positions = torch.from_numpy(
                official_predict.extract_pos_features(relative_target, relative_shape)
            ).unsqueeze(0).to(device)
            source_mask = torch.ones(
                1, len(source), dtype=torch.bool, device=device
            )
            target_mask = torch.ones(
                1, len(target), dtype=torch.bool, device=device
            )
            probability = _edge_probabilities(
                models,
                encoded,
                frame_offset=frame_offset,
                coordinates_source=source_coordinates,
                coordinates_target=target_coordinates,
                positions_source=source_positions,
                positions_target=target_positions,
                mask_source=source_mask,
                mask_target=target_mask,
                downsample_tensor=downsample_tensor,
                config=config,
            )[0].cpu().numpy()
            candidates = sorted(
                (
                    (float(probability[i, j]), i, j)
                    for i in range(len(source))
                    for j in range(len(target))
                    if probability[i, j] > config.edge_threshold
                ),
                reverse=True,
            )
            for score, source_local, target_local in candidates:
                source_global = int(source_indices[source_local])
                target_global = int(target_indices[target_local])
                distance = float(
                    np.linalg.norm(
                        coords_so_far[source_global, 1:].astype(np.float32)
                        - coords_so_far[target_global, 1:].astype(np.float32)
                    )
                )
                edges.append((source_global, target_global, score, distance))
        del encoded

    if seen_frames != set(range(frame_count)):
        raise RuntimeError("linker did not visit every movie frame")
    if seen_pairs != set(zip(range(frame_count - 1), range(1, frame_count))):
        raise RuntimeError("linker did not score every adjacent frame pair")
    association_coords = (
        np.concatenate(coord_lists)
        if coord_lists
        else np.empty((0, 4), dtype=np.int16)
    )
    linked_coords = association_coords.astype(np.float32)
    linked_coords[:, 1:] *= downsample_array
    precise = cache.restore_precise_output(linked_coords)
    manifest = {
        "schema_version": 1,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_source_license": OFFICIAL_SOURCE_LICENSE,
        "association": asdict(config),
        "detector": cache.manifest(),
        "frames": frame_count,
        "nodes": len(precise),
        "edge_candidates": len(edges),
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return precise, edges, manifest
