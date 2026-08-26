from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.spatial import cKDTree


# Frozen preprocessing statistics from royerlab/hoct commit
# 2ccc5040823bc944ab67790abd1f56eea7cd4f05.  The checkpoint consumes
# [t, z, y, x, equivalent diameter, four intensity statistics, the flattened
# 3x3 inertia tensor, border distance].  Biohub detector graphs contain points
# rather than masks, so unavailable morphology channels are set to their
# training means.  They therefore become exactly zero after standardization
# instead of looking like extreme out-of-distribution measurements.
OFFICIAL_FEATURE_MEAN = np.asarray(
    [
        4.6326e02,
        2.9380e00,
        3.5649e02,
        3.4491e02,
        1.1521e01,
        2.7600e-01,
        9.6600e-01,
        5.7400e-01,
        1.6200e-01,
        1.6781e02,
        -2.7000e-02,
        5.0000e-02,
        -2.7000e-02,
        8.7012e01,
        -1.4010e00,
        5.0000e-02,
        -1.4010e00,
        8.3695e01,
        9.0000e-03,
    ],
    dtype=np.float32,
)
OFFICIAL_FEATURE_STD = np.asarray(
    [
        5.5578e02,
        7.6000e00,
        1.9588e02,
        2.2610e02,
        8.1990e00,
        2.1600e-01,
        2.8100e-01,
        1.9300e-01,
        6.9000e-02,
        6.7845e02,
        3.1670e00,
        2.8750e00,
        3.1670e00,
        5.1292e02,
        1.8274e02,
        2.8750e00,
        1.8274e02,
        3.0608e02,
        7.8000e-02,
    ],
    dtype=np.float32,
)
BIOHUB_MODEL_SPATIAL_SCALE = np.asarray((4.0, 1.0, 1.0), dtype=np.float32)


@dataclass(frozen=True)
class HOCTWindow:
    """One point-only temporal window ready for a HOCT forward pass."""

    node_ids: np.ndarray
    times: np.ndarray
    node_features: np.ndarray
    node_positions: np.ndarray
    edge_indices: np.ndarray
    edge_positions: np.ndarray
    edge_delta_t: np.ndarray
    edge_labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_nodes = len(self.node_ids)
        n_edges = len(self.edge_indices)
        if self.node_features.shape != (n_nodes, 19):
            raise ValueError("HOCT node features must have shape (N, 19)")
        if self.node_positions.shape != (n_nodes, 3):
            raise ValueError("HOCT node positions must have shape (N, 3)")
        if self.edge_indices.shape != (n_edges, 2):
            raise ValueError("HOCT edge indices must have shape (E, 2)")
        if self.edge_positions.shape != (n_edges, 3):
            raise ValueError("HOCT edge positions must have shape (E, 3)")
        if self.edge_delta_t.shape != (n_edges,):
            raise ValueError("HOCT edge delta_t must have shape (E,)")
        if self.edge_labels is not None and self.edge_labels.shape != (n_edges,):
            raise ValueError("HOCT edge labels must have shape (E,)")


@dataclass(frozen=True)
class HOCTPrediction:
    edge_logits: np.ndarray
    edge_probabilities: np.ndarray
    edge_features: np.ndarray
    orphan_logits: np.ndarray
    orphan_probabilities: np.ndarray


@dataclass(frozen=True)
class HOCTTile:
    """A spatially bounded pair window and the edges owned by its core."""

    window: HOCTWindow
    core_edge_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.core_edge_mask.shape != (len(self.window.edge_indices),):
            raise ValueError("Tile core edge mask must have shape (E,)")


def point_node_features(
    times: np.ndarray,
    coords_voxel: np.ndarray,
    spatial_scale: np.ndarray = BIOHUB_MODEL_SPATIAL_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct neutral point-only features for an official HOCT checkpoint."""
    times = np.asarray(times, dtype=np.float32).reshape(-1)
    coords = np.asarray(coords_voxel, dtype=np.float32)
    scale = np.asarray(spatial_scale, dtype=np.float32).reshape(3)
    if coords.shape != (len(times), 3):
        raise ValueError("coords_voxel must have shape (N, 3)")
    if not np.isfinite(times).all() or not np.isfinite(coords).all():
        raise ValueError("Point features contain non-finite coordinates")

    positions = coords * scale[None]
    raw = np.broadcast_to(OFFICIAL_FEATURE_MEAN, (len(times), 19)).copy()
    raw[:, 0] = times
    raw[:, 1:4] = positions
    standardized = (raw - OFFICIAL_FEATURE_MEAN[None]) / OFFICIAL_FEATURE_STD[None]
    return standardized.astype(np.float32), positions.astype(np.float32)


def _knn_pairs(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    neighbors: int,
    max_distance: float,
    bidirectional: bool,
) -> set[tuple[int, int]]:
    if len(source_positions) == 0 or len(target_positions) == 0:
        return set()
    pairs: set[tuple[int, int]] = set()

    def query(left: np.ndarray, right: np.ndarray, reverse: bool) -> None:
        k = min(int(neighbors), len(right))
        distances, indices = cKDTree(right).query(left, k=k)
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for left_index, (row_distances, row_indices) in enumerate(
            zip(distances, indices)
        ):
            for distance, right_index in zip(row_distances.tolist(), row_indices.tolist()):
                if float(distance) <= max_distance:
                    pair = (int(right_index), int(left_index)) if reverse else (
                        int(left_index),
                        int(right_index),
                    )
                    pairs.add(pair)

    # Each source proposes children.  The reverse query also guarantees that
    # every target sees its nearest plausible parents, which is important for
    # parental softmax and division recall in unequal-density frames.
    query(source_positions, target_positions, reverse=False)
    if bidirectional:
        query(target_positions, source_positions, reverse=True)
    return pairs


def candidate_edges(
    times: np.ndarray,
    positions: np.ndarray,
    *,
    neighbors: int = 8,
    max_distance: float = 80.0,
    max_delta_t: int = 1,
    bidirectional: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse temporal kNN candidates oriented from earlier to later."""
    times = np.asarray(times, dtype=np.int32).reshape(-1)
    positions = np.asarray(positions, dtype=np.float32)
    if positions.shape != (len(times), 3):
        raise ValueError("positions must have shape (N, 3)")
    if neighbors <= 0 or max_delta_t <= 0 or max_distance <= 0:
        raise ValueError("Candidate graph limits must be positive")

    edges: set[tuple[int, int]] = set()
    delta_by_edge: dict[tuple[int, int], int] = {}
    unique_times = set(int(value) for value in np.unique(times))
    for source_t in sorted(unique_times):
        source_indices = np.flatnonzero(times == source_t)
        for delta_t in range(1, max_delta_t + 1):
            target_t = source_t + delta_t
            if target_t not in unique_times:
                continue
            target_indices = np.flatnonzero(times == target_t)
            local_pairs = _knn_pairs(
                positions[source_indices],
                positions[target_indices],
                neighbors,
                max_distance,
                bidirectional,
            )
            for source_local, target_local in local_pairs:
                edge = (
                    int(source_indices[source_local]),
                    int(target_indices[target_local]),
                )
                edges.add(edge)
                delta_by_edge[edge] = delta_t

    ordered = sorted(edges)
    edge_array = np.asarray(ordered, dtype=np.int64).reshape(-1, 2)
    delta = np.asarray([delta_by_edge[edge] for edge in ordered], dtype=np.float32)
    return edge_array, delta


def build_window(
    node_ids: np.ndarray,
    times: np.ndarray,
    coords_voxel: np.ndarray,
    *,
    start: int,
    window_size: int = 5,
    neighbors: int = 8,
    max_distance: float = 80.0,
    max_delta_t: int = 1,
    true_edges: Iterable[tuple[int, int]] | None = None,
    spatial_scale: np.ndarray = BIOHUB_MODEL_SPATIAL_SCALE,
) -> HOCTWindow:
    """Select a temporal window and construct its HOCT point graph."""
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.int32).reshape(-1)
    coords = np.asarray(coords_voxel, dtype=np.float32)
    if len(node_ids) != len(times) or coords.shape != (len(node_ids), 3):
        raise ValueError("Inconsistent point graph arrays")
    selected = np.flatnonzero((times >= start) & (times < start + window_size))
    selected_ids = node_ids[selected]
    selected_times = times[selected]
    selected_coords = coords[selected]
    features, positions = point_node_features(
        selected_times, selected_coords, spatial_scale=spatial_scale
    )
    edge_indices, edge_delta_t = candidate_edges(
        selected_times,
        positions,
        neighbors=neighbors,
        max_distance=max_distance,
        max_delta_t=max_delta_t,
    )
    edge_positions = (
        (positions[edge_indices[:, 0]] + positions[edge_indices[:, 1]]) * 0.5
        if len(edge_indices)
        else np.empty((0, 3), dtype=np.float32)
    )
    labels = None
    if true_edges is not None:
        truth = {(int(source), int(target)) for source, target in true_edges}
        labels = np.asarray(
            [
                (int(selected_ids[source]), int(selected_ids[target])) in truth
                for source, target in edge_indices
            ],
            dtype=np.float32,
        )
    return HOCTWindow(
        node_ids=selected_ids,
        times=selected_times,
        node_features=features,
        node_positions=positions,
        edge_indices=edge_indices,
        edge_positions=edge_positions,
        edge_delta_t=edge_delta_t,
        edge_labels=labels,
    )


def iter_pair_tiles(
    node_ids: np.ndarray,
    times: np.ndarray,
    coords_voxel: np.ndarray,
    *,
    source_t: int,
    core_size: np.ndarray = np.asarray((128.0, 128.0, 128.0), dtype=np.float32),
    context_halo: float = 80.0,
    neighbors: int = 8,
    max_distance: float = 80.0,
    true_edges: Iterable[tuple[int, int]] | None = None,
    spatial_scale: np.ndarray = BIOHUB_MODEL_SPATIAL_SCALE,
) -> list[HOCTTile]:
    """Tile a dense frame pair while assigning every target to one core.

    Context targets and sources in the halo participate in higher-order edge
    attention.  Only edges ending at a core target are retained, so overlapping
    tiles cannot produce duplicate ownership or inconsistent aggregation.
    """
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.int32).reshape(-1)
    coords = np.asarray(coords_voxel, dtype=np.float32)
    scale = np.asarray(spatial_scale, dtype=np.float32).reshape(3)
    size = np.asarray(core_size, dtype=np.float32).reshape(3)
    if len(node_ids) != len(times) or coords.shape != (len(node_ids), 3):
        raise ValueError("Inconsistent point graph arrays")
    if (size <= 0).any() or context_halo < max_distance:
        raise ValueError("Tile size must be positive and halo must cover candidates")

    source_global = np.flatnonzero(times == source_t)
    target_global = np.flatnonzero(times == source_t + 1)
    if len(source_global) == 0 or len(target_global) == 0:
        return []
    positions = coords * scale[None]
    target_positions = positions[target_global]
    keys = np.floor(target_positions / size[None]).astype(np.int64)
    tiles: list[HOCTTile] = []
    for key in sorted(set(map(tuple, keys.tolist()))):
        key_array = np.asarray(key, dtype=np.int64)
        lower = key_array * size
        upper = lower + size
        core_local_mask = np.all(
            (target_positions >= lower[None]) & (target_positions < upper[None]), axis=1
        )
        core_target_global = target_global[core_local_mask]
        halo_lower = lower - context_halo
        halo_upper = upper + context_halo

        def in_halo(indices: np.ndarray) -> np.ndarray:
            values = positions[indices]
            return indices[
                np.all(
                    (values >= halo_lower[None]) & (values < halo_upper[None]), axis=1
                )
            ]

        selected_global = np.concatenate(
            [in_halo(source_global), in_halo(target_global)]
        )
        window = build_window(
            node_ids[selected_global],
            times[selected_global],
            coords[selected_global],
            start=source_t,
            window_size=2,
            neighbors=neighbors,
            max_distance=max_distance,
            max_delta_t=1,
            true_edges=true_edges,
            spatial_scale=scale,
        )
        if len(window.edge_indices) == 0:
            continue
        core_ids = set(map(int, node_ids[core_target_global].tolist()))
        edge_target_ids = window.node_ids[window.edge_indices[:, 1]]
        core_edge_mask = np.asarray(
            [int(target_id) in core_ids for target_id in edge_target_ids], dtype=bool
        )
        if core_edge_mask.any():
            tiles.append(HOCTTile(window=window, core_edge_mask=core_edge_mask))
    return tiles


def parental_softmax(
    edge_logits: np.ndarray,
    orphan_logits: np.ndarray,
    edge_indices: np.ndarray,
    edge_delta_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize candidate parents together with HOCT's no-parent option."""
    edge_logits = np.asarray(edge_logits, dtype=np.float64).reshape(-1)
    orphan_logits = np.asarray(orphan_logits, dtype=np.float64).reshape(-1)
    edge_indices = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2)
    edge_delta_t = np.asarray(edge_delta_t, dtype=np.float64).reshape(-1)
    if len(edge_logits) != len(edge_indices) or len(edge_delta_t) != len(edge_indices):
        raise ValueError("Inconsistent edge arrays for parental softmax")

    edge_probabilities = np.zeros(len(edge_logits), dtype=np.float32)
    orphan_probability_samples: dict[int, list[float]] = {}
    groups: dict[tuple[int, float], list[int]] = {}
    for edge_index, (target, delta_t) in enumerate(
        zip(edge_indices[:, 1].tolist(), edge_delta_t.tolist())
    ):
        groups.setdefault((int(target), float(delta_t)), []).append(edge_index)

    for (target, _delta_t), indices in groups.items():
        if target < 0 or target >= len(orphan_logits):
            raise ValueError("Edge target is outside orphan logit array")
        values = np.concatenate(
            [edge_logits[indices], np.asarray([orphan_logits[target]])]
        )
        values -= values.max()
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        edge_probabilities[indices] = probabilities[:-1].astype(np.float32)
        orphan_probability_samples.setdefault(target, []).append(float(probabilities[-1]))

    orphan_probabilities = np.ones(len(orphan_logits), dtype=np.float32)
    for target, values in orphan_probability_samples.items():
        orphan_probabilities[target] = float(np.mean(values))
    return edge_probabilities, orphan_probabilities


@torch.inference_mode()
def predict_window(
    model: torch.nn.Module,
    window: HOCTWindow,
    *,
    device: torch.device | str,
    head: torch.nn.Module | None = None,
) -> HOCTPrediction:
    """Run an official scripted HOCT backbone on one Biohub point window."""
    if len(window.edge_indices) == 0:
        raise ValueError("Cannot predict a window with no candidate edges")
    device = torch.device(device)
    node_features = torch.from_numpy(window.node_features).unsqueeze(0).to(device)
    node_positions = torch.from_numpy(window.node_positions).unsqueeze(0).to(device)
    edge_positions = torch.from_numpy(window.edge_positions).unsqueeze(0).to(device)
    edge_indices = torch.from_numpy(window.edge_indices).unsqueeze(0).to(device)
    node_mask = torch.ones((1, len(window.node_ids)), dtype=torch.bool, device=device)
    edge_mask = torch.ones(
        (1, len(window.edge_indices)), dtype=torch.bool, device=device
    )
    output = model.forward(
        node_features,
        node_positions,
        edge_positions,
        edge_indices,
        node_mask,
        edge_mask,
    )
    pretrained_logits, _node_features, edge_features, orphan_logits = output
    # The original classifier applies the frozen head LayerNorm before its
    # linear layer.  Returning that same representation lets a Biohub probe
    # initialize from and remain comparable with the official head.
    if hasattr(model, "head_norm"):
        head_features = model.head_norm(edge_features.float())
    else:
        head_features = edge_features.float()
    if head is None:
        logits = pretrained_logits
    else:
        logits = head(head_features)
    logits_numpy = logits[0, :, 0].float().cpu().numpy()
    edge_features_numpy = head_features[0].float().cpu().numpy()
    orphan_logits_numpy = orphan_logits[0, :, 0].float().cpu().numpy()
    probabilities, orphan_probabilities = parental_softmax(
        logits_numpy,
        orphan_logits_numpy,
        window.edge_indices,
        window.edge_delta_t,
    )
    return HOCTPrediction(
        edge_logits=logits_numpy,
        edge_probabilities=probabilities,
        edge_features=edge_features_numpy,
        orphan_logits=orphan_logits_numpy,
        orphan_probabilities=orphan_probabilities,
    )


def pair_score_matrices(
    window: HOCTWindow,
    probabilities: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Scatter sparse HOCT probabilities into consecutive-frame score matrices."""
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if len(probabilities) != len(window.edge_indices):
        raise ValueError("Probability count does not match window edges")
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for source_t in sorted(int(value) for value in np.unique(window.times)):
        source_local = np.flatnonzero(window.times == source_t)
        target_local = np.flatnonzero(window.times == source_t + 1)
        if len(source_local) == 0 or len(target_local) == 0:
            continue
        source_row = {int(index): row for row, index in enumerate(source_local)}
        target_col = {int(index): col for col, index in enumerate(target_local)}
        matrix = np.zeros((len(source_local), len(target_local)), dtype=np.float32)
        for edge_index, (source, target) in enumerate(window.edge_indices.tolist()):
            row = source_row.get(int(source))
            col = target_col.get(int(target))
            if row is not None and col is not None:
                matrix[row, col] = max(matrix[row, col], probabilities[edge_index])
        result[source_t] = (
            window.node_ids[source_local],
            window.node_ids[target_local],
            matrix,
        )
    return result
