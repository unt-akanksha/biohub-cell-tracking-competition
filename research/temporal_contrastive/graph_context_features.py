"""Node-only, daughter-symmetric context features shared by training and inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


VOXEL_SIZE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
CONTEXT_TIME_OFFSETS = (-2, -1, 0, 1, 2)
CONTEXT_NEIGHBORS_PER_TIME = 8
CONTEXT_RADIUS_UM = 30.0
CONTEXT_SPATIAL_SCALE_UM = 20.0
CONTEXT_TOKEN_COUNT = 43
CONTEXT_FEATURE_WIDTH = 8


def physical_nodes(
    nodes_by_id: dict[int, Any],
) -> dict[int, tuple[int, np.ndarray]]:
    result = {}
    for raw_id, row in nodes_by_id.items():
        node_id = int(raw_id)
        if isinstance(row, dict):
            timepoint = int(row["t"])
            coordinate = np.asarray((row["z"], row["y"], row["x"]), dtype=np.float32)
        else:
            timepoint = int(row[0])
            coordinate = np.asarray(row[1:4], dtype=np.float32)
        result[node_id] = (timepoint, coordinate * VOXEL_SIZE_ZYX_UM)
    return result


def _token(
    *,
    relative_time: int,
    delta_um: np.ndarray,
    parent_anchor: bool,
    daughter_anchor: bool,
    context_node: bool,
) -> np.ndarray:
    distance = float(np.linalg.norm(delta_um))
    return np.asarray(
        (
            relative_time / 2.0,
            *(np.clip(delta_um / CONTEXT_SPATIAL_SCALE_UM, -2.0, 2.0)),
            math.log1p(distance) / math.log1p(CONTEXT_RADIUS_UM),
            float(parent_anchor),
            float(daughter_anchor),
            float(context_node),
        ),
        dtype=np.float32,
    )


def context_tokens(
    nodes: dict[int, tuple[int, np.ndarray]],
    *,
    parent_id: int,
    existing_child_id: int,
    proposed_child_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_ids = {int(parent_id), int(existing_child_id), int(proposed_child_id)}
    if len(anchor_ids) != 3 or not anchor_ids.issubset(nodes):
        raise ValueError("graph-context candidate anchors are invalid")
    parent_time, parent_position = nodes[int(parent_id)]
    if any(
        nodes[node_id][0] != parent_time + 1
        for node_id in (int(existing_child_id), int(proposed_child_id))
    ):
        raise ValueError("graph-context daughters must follow the parent")
    features = np.zeros((CONTEXT_TOKEN_COUNT, CONTEXT_FEATURE_WIDTH), dtype=np.float32)
    mask = np.zeros(CONTEXT_TOKEN_COUNT, dtype=np.bool_)
    features[0] = _token(
        relative_time=0,
        delta_um=np.zeros(3, dtype=np.float32),
        parent_anchor=True,
        daughter_anchor=False,
        context_node=False,
    )
    mask[0] = True
    daughters = sorted(
        (
            (tuple(float(value) for value in nodes[node_id][1]), node_id, nodes[node_id][1])
            for node_id in (int(existing_child_id), int(proposed_child_id))
        ),
        key=lambda row: (row[0], row[1]),
    )
    for index, (_key, _node_id, position) in enumerate(daughters, start=1):
        features[index] = _token(
            relative_time=1,
            delta_um=position - parent_position,
            parent_anchor=False,
            daughter_anchor=True,
            context_node=False,
        )
        mask[index] = True
    by_time: dict[int, list[tuple[int, np.ndarray]]] = {}
    for node_id, (timepoint, position) in nodes.items():
        if node_id not in anchor_ids:
            by_time.setdefault(timepoint, []).append((node_id, position))
    index = 3
    for relative_time in CONTEXT_TIME_OFFSETS:
        rows = []
        for node_id, position in by_time.get(parent_time + relative_time, ()):
            delta = position - parent_position
            distance = float(np.linalg.norm(delta))
            if distance <= CONTEXT_RADIUS_UM:
                rows.append((distance, int(node_id), delta))
        rows.sort(key=lambda row: (row[0], row[1]))
        for distance, _node_id, delta in rows[:CONTEXT_NEIGHBORS_PER_TIME]:
            if not math.isfinite(distance):
                raise ValueError("non-finite graph-context node distance")
            features[index] = _token(
                relative_time=relative_time,
                delta_um=delta,
                parent_anchor=False,
                daughter_anchor=False,
                context_node=True,
            )
            mask[index] = True
            index += 1
        index += CONTEXT_NEIGHBORS_PER_TIME - min(
            len(rows), CONTEXT_NEIGHBORS_PER_TIME
        )
    if index != CONTEXT_TOKEN_COUNT:
        raise RuntimeError("graph-context token layout changed")
    return features, mask
