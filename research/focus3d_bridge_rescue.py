"""Frozen, topology-constrained FOCUS single-frame bridge proposal."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


def _validate_graph(nodes, edges, *, proposal_coordinates=False):
    incoming, outgoing, pairs = {}, {}, set()
    for key, node in nodes.items():
        if int(key) != key or int(node.get("node_id", key)) != int(key):
            raise ValueError("Node identity mismatch")
        values = np.asarray([node[k] for k in ("t", "z", "y", "x")], dtype=float)
        if (not np.isfinite(values).all() or values[0] < 0
                or values[0] != int(values[0])
                or (not proposal_coordinates and (values[1:] < 0).any())):
            raise ValueError("Invalid node time or coordinates")
    for edge in edges:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        if source not in nodes or target not in nodes:
            raise ValueError("Dangling edge")
        if int(nodes[target]["t"]) != int(nodes[source]["t"]) + 1:
            raise ValueError("Edge must join consecutive frames")
        if (source, target) in pairs:
            raise ValueError("Duplicate edge")
        pairs.add((source, target))
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1
        if incoming[target] > 1 or outgoing[source] > 2:
            raise ValueError("Invalid lineage degree")


def _position(node: Mapping[str, Any]) -> np.ndarray:
    return np.asarray((node["z"], node["y"], node["x"]), dtype=np.float64) * SCALE_ZYX_UM


def apply_focus_bridge(
    base_nodes: Mapping[int, Mapping[str, Any]],
    base_edges: Sequence[Mapping[str, Any]],
    focus_nodes: Mapping[int, Mapping[str, Any]],
    focus_edges: Sequence[Mapping[str, Any]],
    deepcenter_probability: Mapping[int, float],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Add at most one FOCUS/DeepCenter-confirmed node and its two edges."""
    output_nodes = {int(key): dict(value) for key, value in base_nodes.items()}
    output_edges = [dict(edge) for edge in base_edges]
    normalized_focus = {int(key): dict(value) for key, value in focus_nodes.items()}
    normalized_probability = {
        int(key): float(value) for key, value in deepcenter_probability.items()
    }
    if len(output_nodes) != len(base_nodes) or len(normalized_focus) != len(focus_nodes):
        raise ValueError("Node IDs collide after normalization")
    base_nodes = output_nodes
    _validate_graph(base_nodes, base_edges)
    # Smoothed proposal graphs can overshoot the image border. Their support
    # topology is still usable, but a negative point must never become a node.
    _validate_graph(normalized_focus, focus_edges, proposal_coordinates=True)
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for edge in base_edges:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        outgoing.setdefault(source, []).append(target)
        incoming.setdefault(target, []).append(source)
    focus_outgoing: dict[int, int] = {}
    focus_incoming: dict[int, int] = {}
    for edge in focus_edges:
        focus_outgoing[int(edge["source_id"])] = focus_outgoing.get(int(edge["source_id"]), 0) + 1
        focus_incoming[int(edge["target_id"])] = focus_incoming.get(int(edge["target_id"]), 0) + 1
    base_by_time: dict[int, list[int]] = {}
    for node_id, node in base_nodes.items():
        base_by_time.setdefault(int(node["t"]), []).append(int(node_id))

    candidates: list[tuple[float, int, int, int, float]] = []
    for focus_id, focus in normalized_focus.items():
        focus_id = int(focus_id)
        if any(float(focus[axis]) < 0 for axis in ('z', 'y', 'x')):
            continue
        if focus_incoming.get(focus_id, 0) == 0 or focus_outgoing.get(focus_id, 0) == 0:
            continue
        probability = float(normalized_probability.get(focus_id, float("nan")))
        if not math.isfinite(probability) or probability < 0.25:
            continue
        timepoint = int(focus["t"])
        focus_position = _position(focus)
        if any(
            np.linalg.norm(focus_position - _position(base_nodes[node_id])) < 3.0
            for node_id in base_by_time.get(timepoint, ())
        ):
            continue
        predecessors = [
            node_id for node_id in base_by_time.get(timepoint - 1, ())
            if not outgoing.get(node_id) and len(incoming.get(node_id, ())) <= 1
        ]
        successors = [
            node_id for node_id in base_by_time.get(timepoint + 1, ())
            if not incoming.get(node_id) and len(outgoing.get(node_id, ())) <= 1
        ]
        for predecessor in predecessors:
            p_position = _position(base_nodes[predecessor])
            p_step = float(np.linalg.norm(focus_position - p_position))
            if p_step > 5.0:
                continue
            for successor in successors:
                s_position = _position(base_nodes[successor])
                s_step = float(np.linalg.norm(s_position - focus_position))
                endpoint_distance = float(np.linalg.norm(s_position - p_position))
                residual = float(np.linalg.norm(focus_position - 0.5 * (p_position + s_position)))
                if s_step <= 5.0 and endpoint_distance <= 10.0 and residual <= 1.5:
                    candidates.append((residual, focus_id, predecessor, successor, endpoint_distance))

    if candidates:
        residual, focus_id, predecessor, successor, _ = min(candidates)
        new_id = max(output_nodes, default=-1) + 1
        focus = normalized_focus[focus_id]
        output_nodes[new_id] = {
            "node_id": new_id,
            "t": int(focus["t"]),
            "z": float(focus["z"]),
            "y": float(focus["y"]),
            "x": float(focus["x"]),
            "focus_bridge_rescue": True,
            "focus_source_node_id": focus_id,
            "deepcenter_probability": float(normalized_probability[focus_id]),
            "interpolation_residual_um": residual,
        }
        output_edges.extend(
            [
                {"source_id": predecessor, "target_id": new_id, "focus_bridge_rescue": True},
                {"source_id": new_id, "target_id": successor, "focus_bridge_rescue": True},
            ]
        )
    _validate_graph(output_nodes, output_edges)
    return output_nodes, output_edges, {
        "eligible_bridges": len(candidates),
        "added_nodes": int(bool(candidates)),
        "added_edges": 2 * int(bool(candidates)),
        "reassignment_performed": 0,
        "existing_node_or_edge_changes": 0,
    }
