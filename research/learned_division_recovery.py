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
    biological_geometry_minimum: float | None = None

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
        if self.biological_geometry_minimum is not None and (
            not math.isfinite(self.biological_geometry_minimum)
            or self.biological_geometry_minimum < 0
        ):
            raise ValueError("division recovery geometry minimum must be finite and nonnegative")


@dataclass(frozen=True)
class DivisionRecoveryCandidate:
    parent_id: int
    existing_child_id: int
    second_child_id: int
    parent_distance_um: float
    sister_distance_um: float
    existing_distance_um: float
    daughter_midpoint_distance_um: float
    daughter_opposition_cosine: float
    daughter_step_ratio: float
    biological_geometry_score: float


def _node_position(node: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        (float(node["z"]), float(node["y"]), float(node["x"])),
        dtype=np.float64,
    )


def biological_geometry_score(
    existing_step_um: np.ndarray, second_step_um: np.ndarray
) -> tuple[float, float, float, float, float]:
    existing_distance = float(np.linalg.norm(existing_step_um))
    second_distance = float(np.linalg.norm(second_step_um))
    denominator = max(existing_distance * second_distance, 1e-12)
    opposition_cosine = float(
        np.dot(existing_step_um, second_step_um) / denominator
    )
    midpoint_distance = float(
        np.linalg.norm(0.5 * (existing_step_um + second_step_um))
    )
    step_ratio = float(
        min(existing_distance, second_distance)
        / max(existing_distance, second_distance, 1e-12)
    )
    score = float(
        min(existing_distance, second_distance)
        * step_ratio
        * (1.0 - opposition_cosine)
        / (1.0 + midpoint_distance / 4.0)
    )
    return (
        score,
        existing_distance,
        midpoint_distance,
        opposition_cosine,
        step_ratio,
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
        existing_step = sister_position - parent_position
        eligible: list[tuple[float, float, int, tuple[float, ...]]] = []
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
                geometry = biological_geometry_score(
                    existing_step, position - parent_position
                )
                eligible.append((parent_distance, sister_distance, node_id, geometry))
        if eligible:
            parent_distance, sister_distance, second_child_id, geometry = min(eligible)
            candidates.append(
                DivisionRecoveryCandidate(
                    parent_id=parent_id,
                    existing_child_id=existing_child_id,
                    second_child_id=second_child_id,
                    parent_distance_um=parent_distance,
                    sister_distance_um=sister_distance,
                    existing_distance_um=geometry[1],
                    daughter_midpoint_distance_um=geometry[2],
                    daughter_opposition_cosine=geometry[3],
                    daughter_step_ratio=geometry[4],
                    biological_geometry_score=geometry[0],
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
    geometry_rejected = 0
    for candidate in candidates:
        if candidate.parent_id not in division_logits:
            missing_scores.append(candidate.parent_id)
            continue
        score = float(division_logits[candidate.parent_id])
        if not math.isfinite(score):
            raise ValueError("division recovery logits must be finite")
        if (
            policy.biological_geometry_minimum is not None
            and candidate.biological_geometry_score
            < policy.biological_geometry_minimum
        ):
            geometry_rejected += 1
            continue
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
                "biological_geometry_score": candidate.biological_geometry_score,
                "learned_division_recovery": True,
            }
        )
        existing_pairs.add(pair)
    stats = {
        "geometric_candidates": len(candidates),
        "parents_missing_learned_score": len(missing_scores),
        "learned_gate_candidates": len(scored),
        "biological_geometry_minimum": policy.biological_geometry_minimum,
        "geometry_rejected": geometry_rejected,
        "maximum_additions": maximum_additions,
        "added_edges": len(selected),
        "cap_rejected": max(0, len(scored) - len(selected)),
        "reassignment_performed": 0,
        "node_or_coordinate_changes": 0,
    }
    return output, stats


def apply_ranked_consensus_division_recovery(
    nodes_by_id: Mapping[int, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    deep_scores: Mapping[int, float],
    morphology_scores: Mapping[int, float],
    *,
    biological_geometry_minimum: float = 3.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add at most one edge when two scale-invariant rankings agree."""
    if not math.isfinite(biological_geometry_minimum) or biological_geometry_minimum < 0:
        raise ValueError("ranked consensus geometry minimum must be finite and nonnegative")
    candidates = discover_division_recovery_candidates(nodes_by_id, edges)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.biological_geometry_score >= biological_geometry_minimum
        and candidate.parent_id in deep_scores
        and candidate.parent_id in morphology_scores
    ]
    for candidate in eligible:
        if not (
            math.isfinite(float(deep_scores[candidate.parent_id]))
            and math.isfinite(float(morphology_scores[candidate.parent_id]))
        ):
            raise ValueError("ranked consensus scores must be finite")
    selected: DivisionRecoveryCandidate | None = None
    if eligible:
        deep_top = max(
            eligible,
            key=lambda candidate: (
                float(deep_scores[candidate.parent_id]),
                -candidate.parent_distance_um,
                -candidate.sister_distance_um,
                -candidate.parent_id,
            ),
        )
        morphology_top = max(
            eligible,
            key=lambda candidate: (
                float(morphology_scores[candidate.parent_id]),
                -candidate.parent_distance_um,
                -candidate.sister_distance_um,
                -candidate.parent_id,
            ),
        )
        if deep_top.parent_id == morphology_top.parent_id:
            selected = deep_top
    output = [dict(edge) for edge in edges]
    if selected is not None:
        pair = (selected.parent_id, selected.second_child_id)
        existing_pairs = {
            (int(edge["source_id"]), int(edge["target_id"])) for edge in output
        }
        if pair in existing_pairs:
            raise RuntimeError("ranked consensus selected an existing edge")
        output.append(
            {
                "source_id": selected.parent_id,
                "target_id": selected.second_child_id,
                "distance_um": selected.parent_distance_um,
                "deep_division_score": float(deep_scores[selected.parent_id]),
                "morphology_division_score": float(
                    morphology_scores[selected.parent_id]
                ),
                "biological_geometry_score": selected.biological_geometry_score,
                "ranked_consensus_division_recovery": True,
            }
        )
    return output, {
        "geometric_candidates": len(candidates),
        "geometry_eligible_candidates": len(eligible),
        "candidate_parents_scored": len(set(deep_scores) & set(morphology_scores)),
        "ranking_agreed": selected is not None,
        "added_edges": int(selected is not None),
        "maximum_additions": 1,
        "biological_geometry_minimum": biological_geometry_minimum,
        "absolute_threshold_used": False,
        "reassignment_performed": 0,
        "node_or_coordinate_changes": 0,
    }
