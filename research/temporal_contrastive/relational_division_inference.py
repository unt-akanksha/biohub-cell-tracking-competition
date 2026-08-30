"""Candidate-aligned inference for heavy relational division models."""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

try:
    from learned_division_recovery import (
        VOXEL_SIZE_ZYX_UM,
        DivisionRecoveryCandidate,
    )
except ModuleNotFoundError:
    from research.learned_division_recovery import (
        VOXEL_SIZE_ZYX_UM,
        DivisionRecoveryCandidate,
    )


GEOMETRY_NAMES = (
    "parent_distance_um",
    "sister_distance_um",
    "existing_distance_um",
    "daughter_midpoint_distance_um",
    "daughter_opposition_cosine",
    "daughter_step_ratio",
    "biological_geometry_score",
    "parent_velocity_um",
    "constant_velocity_midpoint_error_um",
)


def _position(node: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        (float(node["z"]), float(node["y"]), float(node["x"])),
        dtype=np.float64,
    )


def candidate_centers_zyx(
    candidate: DivisionRecoveryCandidate,
    nodes_by_id: Mapping[int, Mapping[str, Any]],
) -> np.ndarray:
    identifiers = (
        candidate.parent_id,
        candidate.existing_child_id,
        candidate.second_child_id,
    )
    if any(identifier not in nodes_by_id for identifier in identifiers):
        raise ValueError("relational candidate refers to a missing node")
    parent_time = int(nodes_by_id[candidate.parent_id]["t"])
    if any(
        int(nodes_by_id[identifier]["t"]) != parent_time + 1
        for identifier in identifiers[1:]
    ):
        raise ValueError("relational candidate daughters must be in the next frame")
    return np.stack([_position(nodes_by_id[identifier]) for identifier in identifiers])


def candidate_geometry_features(
    candidate: DivisionRecoveryCandidate,
    nodes_by_id: Mapping[int, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    centers = candidate_centers_zyx(candidate, nodes_by_id)
    physical = centers * VOXEL_SIZE_ZYX_UM[None, :]
    incoming = [
        int(edge["source_id"])
        for edge in edges
        if int(edge["target_id"]) == candidate.parent_id
        and int(edge["source_id"]) in nodes_by_id
        and int(nodes_by_id[int(edge["source_id"])]["t"])
        == int(nodes_by_id[candidate.parent_id]["t"]) - 1
    ]
    velocity = None
    if len(set(incoming)) == 1:
        velocity = physical[0] - (
            _position(nodes_by_id[incoming[0]]) * VOXEL_SIZE_ZYX_UM
        )
    midpoint = 0.5 * (physical[1] + physical[2])
    values = np.asarray(
        (
            candidate.parent_distance_um,
            candidate.sister_distance_um,
            candidate.existing_distance_um,
            candidate.daughter_midpoint_distance_um,
            candidate.daughter_opposition_cosine,
            candidate.daughter_step_ratio,
            candidate.biological_geometry_score,
            float(np.linalg.norm(velocity)) if velocity is not None else math.nan,
            (
                float(np.linalg.norm(midpoint - (physical[0] + velocity)))
                if velocity is not None
                else math.nan
            ),
        ),
        dtype=np.float32,
    )
    if values.shape != (len(GEOMETRY_NAMES),):
        raise RuntimeError("relational inference geometry width changed")
    return values


def calibration_free_parent_scores(
    member_scores: Sequence[Mapping[int, float]],
    parent_ids: Sequence[int],
) -> dict[int, float]:
    parents = [int(value) for value in parent_ids]
    if not member_scores or not parents or len(set(parents)) != len(parents):
        raise ValueError("relational rank consensus requires unique parents and members")
    rank_matrix = np.empty((len(member_scores), len(parents)), dtype=np.float64)
    for member_index, scores in enumerate(member_scores):
        if set(scores) != set(parents):
            raise ValueError("relational member scores are not aligned")
        raw = np.asarray([float(scores[parent]) for parent in parents], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("relational member scores must be finite")
        order = np.argsort(-raw, kind="stable")
        percentiles = (
            np.ones(1, dtype=np.float64)
            if len(parents) == 1
            else np.linspace(1.0, 0.0, len(parents), dtype=np.float64)
        )
        rank_matrix[member_index, order] = percentiles
    consensus = rank_matrix.mean(axis=0)
    return {parent: float(consensus[index]) for index, parent in enumerate(parents)}


@torch.inference_mode()
def score_relational_candidates(
    models: Sequence[torch.nn.Module],
    candidates: Sequence[DivisionRecoveryCandidate],
    nodes_by_id: Mapping[int, Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    read_frame: Callable[[int], np.ndarray],
    sample_physical_patches: Callable[..., torch.Tensor],
    device: torch.device,
    maximum_time: int,
    batch_size: int = 12,
) -> tuple[dict[int, float], dict[str, Any]]:
    if not models or batch_size <= 0 or maximum_time < 0:
        raise ValueError("invalid relational inference configuration")
    parents = [int(candidate.parent_id) for candidate in candidates]
    if len(set(parents)) != len(parents):
        raise ValueError("relational inference requires one candidate per parent")
    if not candidates:
        return {}, {
            "candidate_parents_scored": 0,
            "candidate_frames_scored": 0,
            "relational_member_count": len(models),
            "absolute_threshold_used": False,
        }
    by_time: dict[int, list[DivisionRecoveryCandidate]] = {}
    for candidate in candidates:
        timepoint = int(nodes_by_id[candidate.parent_id]["t"])
        by_time.setdefault(timepoint, []).append(candidate)
    raw_member_scores = [dict() for _ in models]
    for timepoint, frame_candidates in sorted(by_time.items()):
        frame_indices = [
            max(0, min(maximum_time, timepoint + offset))
            for offset in (-1, 0, 1)
        ]
        temporal = np.stack([read_frame(index) for index in frame_indices], axis=0)
        centers = np.concatenate(
            [candidate_centers_zyx(candidate, nodes_by_id) for candidate in frame_candidates],
            axis=0,
        ).astype(np.float32)
        sampled = sample_physical_patches(
            torch.as_tensor(temporal, device=device),
            centers,
            voxel_size_zyx_um=tuple(float(value) for value in VOXEL_SIZE_ZYX_UM),
            chunk_size=max(3, batch_size * 3),
        )
        expected_shape = (len(frame_candidates) * 3, 3, 17, 17, 17)
        if tuple(sampled.shape) != expected_shape:
            raise ValueError(f"relational sampled patch shape changed: {tuple(sampled.shape)}")
        patches = sampled.reshape(len(frame_candidates), 3, 3, 17, 17, 17)
        geometry = torch.as_tensor(
            np.stack(
                [
                    candidate_geometry_features(candidate, nodes_by_id, edges)
                    for candidate in frame_candidates
                ]
            ),
            device=device,
        )
        for member_index, model in enumerate(models):
            pieces = []
            for start in range(0, len(patches), batch_size):
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with context:
                    pieces.append(
                        model(
                            patches[start : start + batch_size],
                            geometry[start : start + batch_size],
                        )
                        .float()
                        .cpu()
                    )
            values = torch.cat(pieces).numpy()
            for candidate, value in zip(frame_candidates, values, strict=True):
                raw_member_scores[member_index][candidate.parent_id] = float(value)
        del sampled, patches, geometry
    consensus = calibration_free_parent_scores(raw_member_scores, sorted(parents))
    return consensus, {
        "candidate_parents_scored": len(parents),
        "candidate_frames_scored": len(by_time),
        "relational_member_count": len(models),
        "absolute_threshold_used": False,
    }
