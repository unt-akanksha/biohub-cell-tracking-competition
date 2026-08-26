from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SwapConfig:
    min_appearance_gain: float
    max_pair_distance_um: float = 12.0
    max_total_distance_increase_um: float = 2.0
    base_lock_probability: float = 0.95
    fallback_edge_probability: float = 0.80

    def __post_init__(self) -> None:
        if self.min_appearance_gain <= 0:
            raise ValueError("min_appearance_gain must be positive")
        if self.max_pair_distance_um <= 0:
            raise ValueError("max_pair_distance_um must be positive")
        if self.max_total_distance_increase_um < 0:
            raise ValueError("max_total_distance_increase_um cannot be negative")
        if not 0 < self.base_lock_probability <= 1:
            raise ValueError("base_lock_probability must be in (0, 1]")
        if not 0 <= self.fallback_edge_probability <= 1:
            raise ValueError("fallback_edge_probability must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _validate_inputs(
    node_ids: np.ndarray,
    times: np.ndarray,
    coords_zyx: np.ndarray,
    embeddings: np.ndarray,
    edges: np.ndarray,
    edge_probabilities: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    node_times = np.asarray(times, dtype=np.int64).reshape(-1)
    coords = np.asarray(coords_zyx, dtype=np.float32)
    features = np.asarray(embeddings, dtype=np.float32)
    base_edges = np.asarray(edges, dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("node_ids must be unique")
    if node_times.shape != ids.shape or coords.shape != (len(ids), 3):
        raise ValueError("node times and coordinates must align with node_ids")
    if features.ndim != 2 or features.shape[0] != len(ids):
        raise ValueError("embeddings must have shape (N, C)")
    if base_edges.ndim != 2 or base_edges.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    if not np.isfinite(coords).all() or not np.isfinite(features).all():
        raise ValueError("coordinates and embeddings must be finite")
    if edge_probabilities is None:
        probabilities = np.full(len(base_edges), np.nan, dtype=np.float32)
    else:
        probabilities = np.asarray(edge_probabilities, dtype=np.float32).reshape(-1)
        if probabilities.shape != (len(base_edges),):
            raise ValueError("edge_probabilities must align with edges")
        if np.any(np.isfinite(probabilities) & ((probabilities < 0) | (probabilities > 1))):
            raise ValueError("finite edge probabilities must be in [0, 1]")
    return ids, node_times, coords, features, base_edges, probabilities


def appearance_pair_swaps(
    node_ids: np.ndarray,
    times: np.ndarray,
    coords_zyx: np.ndarray,
    embeddings: np.ndarray,
    edges: np.ndarray,
    *,
    config: SwapConfig,
    edge_probabilities: np.ndarray | None = None,
    voxel_scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> tuple[list[tuple[int, int]], dict]:
    """Swap disjoint pairs of ambiguous base links using appearance continuity.

    Only two ordinary one-parent/one-child edges from the same frame transition
    can be exchanged. The operation therefore preserves the complete node set,
    total edge count, and every node's in/out degree, including all divisions.
    """
    ids, node_times, coords, features, base_edges, probabilities = _validate_inputs(
        node_ids,
        times,
        coords_zyx,
        embeddings,
        edges,
        edge_probabilities,
    )
    id_to_index = {int(node_id): index for index, node_id in enumerate(ids.tolist())}
    if any(int(node) not in id_to_index for node in base_edges.reshape(-1).tolist()):
        raise ValueError("edges reference unavailable nodes")
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for source, target in base_edges.tolist():
        outgoing[int(source)] = outgoing.get(int(source), 0) + 1
        incoming[int(target)] = incoming.get(int(target), 0) + 1

    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    normalized = features / norms
    scale = np.asarray(voxel_scale_um, dtype=np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("voxel_scale_um must contain three positive finite values")

    eligible_by_time: dict[int, list[int]] = {}
    skipped_nonconsecutive = skipped_degree = skipped_locked = 0
    effective_probabilities = np.where(
        np.isfinite(probabilities), probabilities, config.fallback_edge_probability
    )
    for edge_index, (source, target) in enumerate(base_edges.tolist()):
        source = int(source)
        target = int(target)
        source_index = id_to_index[source]
        target_index = id_to_index[target]
        if node_times[target_index] != node_times[source_index] + 1:
            skipped_nonconsecutive += 1
            continue
        if outgoing[source] != 1 or incoming[target] != 1:
            skipped_degree += 1
            continue
        if effective_probabilities[edge_index] >= config.base_lock_probability:
            skipped_locked += 1
            continue
        eligible_by_time.setdefault(int(node_times[source_index]), []).append(edge_index)

    proposals: list[dict] = []
    candidate_pairs = 0
    for frame, edge_indices in sorted(eligible_by_time.items()):
        if len(edge_indices) < 2:
            continue
        edge_indices_array = np.asarray(edge_indices, dtype=np.int64)
        frame_edges = base_edges[edge_indices_array]
        source_rows = np.asarray(
            [id_to_index[int(source)] for source in frame_edges[:, 0]], dtype=np.int64
        )
        target_rows = np.asarray(
            [id_to_index[int(target)] for target in frame_edges[:, 1]], dtype=np.int64
        )
        source_positions = coords[source_rows] * scale
        target_positions = coords[target_rows] * scale
        distances = np.linalg.norm(
            source_positions[:, None, :] - target_positions[None, :, :], axis=2
        )
        similarities = normalized[source_rows] @ normalized[target_rows].T
        base_distance = np.diag(distances)
        base_similarity = np.diag(similarities)
        for left, right in zip(
            *np.where(
                np.triu(
                    (distances <= config.max_pair_distance_um)
                    & (distances.T <= config.max_pair_distance_um),
                    k=1,
                )
            )
        ):
            candidate_pairs += 1
            distance_increase = float(
                distances[left, right]
                + distances[right, left]
                - base_distance[left]
                - base_distance[right]
            )
            if distance_increase > config.max_total_distance_increase_um:
                continue
            appearance_gain = float(
                similarities[left, right]
                + similarities[right, left]
                - base_similarity[left]
                - base_similarity[right]
            )
            if appearance_gain < config.min_appearance_gain:
                continue
            proposals.append(
                {
                    "frame": frame,
                    "left_edge_index": int(edge_indices_array[left]),
                    "right_edge_index": int(edge_indices_array[right]),
                    "appearance_gain": appearance_gain,
                    "distance_increase_um": distance_increase,
                }
            )

    proposals.sort(
        key=lambda row: (
            -float(row["appearance_gain"]),
            float(row["distance_increase_um"]),
            int(row["frame"]),
            int(row["left_edge_index"]),
            int(row["right_edge_index"]),
        )
    )
    selected: list[dict] = []
    used_edges: set[int] = set()
    output = base_edges.copy()
    for proposal in proposals:
        left = int(proposal["left_edge_index"])
        right = int(proposal["right_edge_index"])
        if left in used_edges or right in used_edges:
            continue
        source_left, target_left = map(int, output[left])
        source_right, target_right = map(int, output[right])
        output[left] = (source_left, target_right)
        output[right] = (source_right, target_left)
        used_edges.update((left, right))
        selected.append(proposal)

    output_outgoing: dict[int, int] = {}
    output_incoming: dict[int, int] = {}
    for source, target in output.tolist():
        output_outgoing[int(source)] = output_outgoing.get(int(source), 0) + 1
        output_incoming[int(target)] = output_incoming.get(int(target), 0) + 1
    if outgoing != output_outgoing or incoming != output_incoming:
        raise RuntimeError("appearance swaps unexpectedly changed graph degree")

    diagnostics = {
        "config": config.to_dict(),
        "base_edges": int(len(base_edges)),
        "eligible_edges": int(sum(map(len, eligible_by_time.values()))),
        "candidate_pairs": int(candidate_pairs),
        "qualified_pairs": int(len(proposals)),
        "selected_swaps": int(len(selected)),
        "changed_edges": int(2 * len(selected)),
        "skipped_nonconsecutive_edges": int(skipped_nonconsecutive),
        "skipped_degree_edges": int(skipped_degree),
        "skipped_locked_edges": int(skipped_locked),
        "selected": selected,
    }
    return sorted((int(source), int(target)) for source, target in output.tolist()), diagnostics
