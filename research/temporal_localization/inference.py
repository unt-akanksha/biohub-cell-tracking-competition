"""Physical-grid inference helpers shared by development and Kaggle runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from research.temporal_contrastive.patch_model import sample_physical_patches
from research.temporal_localization.model import HALF_EXTENT_UM, PATCH_SHAPE, TemporalNodeLocalizationModel


@dataclass(frozen=True)
class MovieGraphArrays:
    node_ids: np.ndarray
    times: np.ndarray
    coordinates_zyx_voxel: np.ndarray
    edges_by_node_id: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.node_ids)
        if self.node_ids.shape != (count,) or self.times.shape != (count,):
            raise ValueError("node identifiers and times must be vectors")
        if self.coordinates_zyx_voxel.shape != (count, 3):
            raise ValueError("node coordinates must have shape (N, 3)")
        if self.edges_by_node_id.ndim != 2 or self.edges_by_node_id.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2)")
        if len(np.unique(self.node_ids)) != count:
            raise ValueError("node identifiers must be unique")
        if not np.isfinite(self.coordinates_zyx_voxel).all():
            raise ValueError("node coordinates must be finite")


def graph_motion_features(
    graph: MovieGraphArrays,
    rows: np.ndarray,
    *,
    voxel_size_zyx_um: np.ndarray,
    spatial_shape_zyx: tuple[int, int, int],
    timepoints: int,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    scale = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    if np.any(rows < 0) or np.any(rows >= len(graph.node_ids)):
        raise ValueError("inference node row is outside the graph")
    if scale.shape != (3,) or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("voxel scale must contain three positive finite values")
    if len(spatial_shape_zyx) != 3 or min(spatial_shape_zyx) <= 0 or timepoints <= 1:
        raise ValueError("movie shape and timepoints must be positive")
    by_id = {int(node_id): row for row, node_id in enumerate(graph.node_ids.tolist())}
    predecessor = np.full(len(graph.node_ids), -1, dtype=np.int64)
    successors: list[list[int]] = [[] for _ in graph.node_ids]
    for source_id, target_id in graph.edges_by_node_id.tolist():
        if int(source_id) not in by_id or int(target_id) not in by_id:
            raise ValueError("graph edge references an unavailable node")
        source, target = by_id[int(source_id)], by_id[int(target_id)]
        if predecessor[target] >= 0 and predecessor[target] != source:
            raise ValueError("graph contains a merge")
        predecessor[target] = source
        successors[source].append(target)
    coords = graph.coordinates_zyx_voxel
    shape = np.asarray(spatial_shape_zyx, dtype=np.float32)
    output = np.zeros((len(rows), 12), dtype=np.float32)
    for batch_row, node_row in enumerate(rows.tolist()):
        proposal = coords[node_row]
        parent = int(predecessor[node_row])
        children = np.asarray(successors[node_row], dtype=np.int64)
        if parent >= 0:
            output[batch_row, 0:3] = (proposal - coords[parent]) * scale / 10.0
            output[batch_row, 6] = 1.0
        if len(children):
            output[batch_row, 3:6] = (coords[children].mean(axis=0) - proposal) * scale / 10.0
            output[batch_row, 7] = 1.0
        output[batch_row, 8] = min(len(children), 2) / 2.0
        same_frame = np.flatnonzero(graph.times == graph.times[node_row])
        same_frame = same_frame[same_frame != node_row]
        if len(same_frame):
            distances = np.linalg.norm((coords[same_frame] - proposal[None]) * scale[None], axis=1)
            output[batch_row, 9] = min(float(distances.min()) / 20.0, 2.0)
        output[batch_row, 10] = float(graph.times[node_row]) / float(timepoints - 1)
        boundary_um = np.minimum(proposal, shape - 1 - proposal) * scale
        output[batch_row, 11] = float(np.clip(boundary_um.min() / 10.0, 0.0, 2.0))
    return np.clip(output, -3.0, 3.0)


@torch.inference_mode()
def predict_member(
    model: TemporalNodeLocalizationModel,
    graph: MovieGraphArrays,
    frame_reader: Callable[[int], np.ndarray],
    frames: Iterable[int],
    *,
    device: torch.device,
    voxel_size_zyx_um: np.ndarray,
    spatial_shape_zyx: tuple[int, int, int],
    timepoints: int,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict one member at specified interior frames in graph-row order."""

    if batch_size <= 0:
        raise ValueError("inference batch size must be positive")
    ordered_frames = tuple(sorted(set(int(value) for value in frames)))
    if any(value <= 0 or value >= timepoints - 1 for value in ordered_frames):
        raise ValueError("temporal localization requires interior frames")
    selected_rows = np.concatenate(
        [np.flatnonzero(graph.times == frame).astype(np.int64) for frame in ordered_frames]
    ) if ordered_frames else np.empty((0,), dtype=np.int64)
    offsets = np.empty((len(selected_rows), 3), dtype=np.float32)
    sigma = np.empty((len(selected_rows), 3), dtype=np.float32)
    safe = np.empty((len(selected_rows),), dtype=np.float32)
    model.eval()
    cursor = 0
    for frame in ordered_frames:
        frame_rows = np.flatnonzero(graph.times == frame).astype(np.int64)
        if not len(frame_rows):
            continue
        context = torch.as_tensor(
            np.stack((frame_reader(frame - 1), frame_reader(frame), frame_reader(frame + 1))),
            dtype=torch.float32,
            device=device,
        )
        features = graph_motion_features(
            graph,
            frame_rows,
            voxel_size_zyx_um=voxel_size_zyx_um,
            spatial_shape_zyx=spatial_shape_zyx,
            timepoints=timepoints,
        )
        for start in range(0, len(frame_rows), batch_size):
            batch_rows = frame_rows[start : start + batch_size]
            patches = sample_physical_patches(
                context,
                torch.as_tensor(
                    graph.coordinates_zyx_voxel[batch_rows], dtype=torch.float32, device=device
                ),
                voxel_size_zyx_um=voxel_size_zyx_um,
                output_shape_zyx=PATCH_SHAPE,
                half_extent_zyx_um=HALF_EXTENT_UM,
                chunk_size=len(batch_rows),
            )
            graph_features = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predicted, log_variance, predicted_safe = model(patches, graph_features)
            stop = cursor + len(batch_rows)
            offsets[cursor:stop] = predicted.float().cpu().numpy()
            sigma[cursor:stop] = torch.exp(0.5 * log_variance.float()).cpu().numpy()
            safe[cursor:stop] = predicted_safe.float().cpu().numpy()
            cursor = stop
    if cursor != len(selected_rows):
        raise RuntimeError("temporal localization inference lost node rows")
    return selected_rows, offsets, sigma, safe
