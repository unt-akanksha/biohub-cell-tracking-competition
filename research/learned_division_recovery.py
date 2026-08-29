"""Constrained second-daughter recovery gated by an external learned model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


VOXEL_SIZE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)


@dataclass(frozen=True)
class DivisionRecoveryPolicy:
    division_logit_threshold: float
    parent_distance_max_um: float = 12.0
    sister_distance_max_um: float = 15.0
    maximum_added_node_fraction: float = 0.0025

    def __post_init__(self) -> None:
        values = (
            self.division_logit_threshold,
            self.parent_distance_max_um,
            self.sister_distance_max_um,
            self.maximum_added_node_fraction,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("division recovery policy values must be finite")
        if self.parent_distance_max_um <= 0 or self.sister_distance_max_um <= 0:
            raise ValueError("division recovery distance limits must be positive")
        if not 0 < self.maximum_added_node_fraction <= 0.01:
            raise ValueError("division recovery cap must be in (0, 0.01]")


@dataclass(frozen=True)
class DivisionRecoveryCandidate:
    parent_id: int
    existing_child_id: int
    second_child_id: int
    parent_distance_um: float
    sister_distance_um: float


def _node_position(node: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        (float(node["z"]), float(node["y"]), float(node["x"])),
        dtype=np.float64,
    )


def discover_division_recovery_candidates(
    nodes_by_id: Mapping[int, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    parent_distance_max_um: float = 12.0,
    sister_distance_max_um: float = 15.0,
) -> list[DivisionRecoveryCandidate]:
    if parent_distance_max_um <= 0 or sister_distance_max_um <= 0:
        raise ValueError("division recovery distances must be positive")
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        if source not in nodes_by_id or target not in nodes_by_id:
            raise ValueError("division recovery received a dangling edge")
        outgoing.setdefault(source, []).append(target)
        incoming.setdefault(target, []).append(source)
    by_time: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        by_time.setdefault(int(node["t"]), []).append(int(node_id))
    candidates: list[DivisionRecoveryCandidate] = []
    for parent_id, children in sorted(outgoing.items()):
        if len(set(children)) != 1:
            continue
        existing_child_id = int(children[0])
        parent = nodes_by_id[parent_id]
        existing = nodes_by_id[existing_child_id]
        next_time = int(parent["t"]) + 1
        if int(existing["t"]) != next_time:
            continue
        parent_position = _node_position(parent) * VOXEL_SIZE_ZYX_UM
        sister_position = _node_position(existing) * VOXEL_SIZE_ZYX_UM
        eligible: list[tuple[float, float, int]] = []
        for node_id in by_time.get(next_time, ()):
            if node_id == existing_child_id or incoming.get(node_id):
                continue
            position = _node_position(nodes_by_id[node_id]) * VOXEL_SIZE_ZYX_UM
            parent_distance = float(np.linalg.norm(position - parent_position))
            sister_distance = float(np.linalg.norm(position - sister_position))
            if (
                parent_distance <= parent_distance_max_um
                and sister_distance <= sister_distance_max_um
            ):
                eligible.append((parent_distance, sister_distance, node_id))
        if eligible:
            parent_distance, sister_distance, second_child_id = min(eligible)
            candidates.append(
                DivisionRecoveryCandidate(
                    parent_id=parent_id,
                    existing_child_id=existing_child_id,
                    second_child_id=second_child_id,
                    parent_distance_um=parent_distance,
                    sister_distance_um=sister_distance,
                )
            )
    return candidates


def apply_learned_division_recovery(
    nodes_by_id: Mapping[int, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    division_logits: Mapping[int, float],
    policy: DivisionRecoveryPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = discover_division_recovery_candidates(
        nodes_by_id,
        edges,
        parent_distance_max_um=policy.parent_distance_max_um,
        sister_distance_max_um=policy.sister_distance_max_um,
    )
    scored: list[tuple[float, DivisionRecoveryCandidate]] = []
    missing_scores: list[int] = []
    for candidate in candidates:
        if candidate.parent_id not in division_logits:
            missing_scores.append(candidate.parent_id)
            continue
        score = float(division_logits[candidate.parent_id])
        if not math.isfinite(score):
            raise ValueError("division recovery logits must be finite")
        if score >= policy.division_logit_threshold:
            scored.append((score, candidate))
    maximum_additions = max(
        1,
        int(math.ceil(len(nodes_by_id) * policy.maximum_added_node_fraction)),
    )
    selected = sorted(
        scored,
        key=lambda row: (
            -row[0],
            row[1].parent_distance_um,
            row[1].sister_distance_um,
            row[1].parent_id,
        ),
    )[:maximum_additions]
    output = [dict(edge) for edge in edges]
    existing_pairs = {
        (int(edge["source_id"]), int(edge["target_id"])) for edge in output
    }
    for score, candidate in selected:
        pair = (candidate.parent_id, candidate.second_child_id)
        if pair in existing_pairs:
            raise RuntimeError("division recovery selected an existing edge")
        output.append(
            {
                "source_id": candidate.parent_id,
                "target_id": candidate.second_child_id,
                "distance_um": candidate.parent_distance_um,
                "division_logit": score,
                "learned_division_recovery": True,
            }
        )
        existing_pairs.add(pair)
    stats = {
        "geometric_candidates": len(candidates),
        "parents_missing_learned_score": len(missing_scores),
        "learned_gate_candidates": len(scored),
        "maximum_additions": maximum_additions,
        "added_edges": len(selected),
        "cap_rejected": max(0, len(scored) - len(selected)),
        "reassignment_performed": 0,
        "node_or_coordinate_changes": 0,
    }
    return output, stats
