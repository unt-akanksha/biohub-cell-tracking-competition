"""Precommitted safety policy for temporal localization ensembles.

Synthetic gates admit models; this module decides whether their independent
predictions agree strongly enough to donate coordinates.  The policy contains
no labels, metric calls, public predictions, or leaderboard-dependent values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from biohub_tracker.graphs import GraphData


NATIVE_VOXEL_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)


@dataclass(frozen=True)
class LocalizationConsensusPolicy:
    minimum_members: int = 3
    blend: float = 0.75
    minimum_correction_um: float = 2.0
    maximum_correction_um: float = 9.5
    maximum_member_disagreement_um: float = 1.5
    maximum_predicted_sigma_um: float = 2.5
    maximum_safe_probability: float = 0.35
    minimum_direction_cosine: float = 0.80
    maximum_move_fraction: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.blend,
            self.minimum_correction_um,
            self.maximum_correction_um,
            self.maximum_member_disagreement_um,
            self.maximum_predicted_sigma_um,
            self.maximum_safe_probability,
            self.minimum_direction_cosine,
            self.maximum_move_fraction,
        )
        if self.minimum_members < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("localization consensus policy values must be finite")
        if not 0.0 < self.blend <= 1.0:
            raise ValueError("localization blend must lie in (0, 1]")
        if not 0.0 < self.minimum_correction_um < self.maximum_correction_um:
            raise ValueError("localization correction bounds are invalid")
        if min(self.maximum_member_disagreement_um, self.maximum_predicted_sigma_um) <= 0:
            raise ValueError("localization uncertainty bounds must be positive")
        if not 0.0 <= self.maximum_safe_probability <= 1.0:
            raise ValueError("safe probability bound must lie in [0, 1]")
        if not -1.0 <= self.minimum_direction_cosine <= 1.0:
            raise ValueError("direction cosine bound must lie in [-1, 1]")
        if not 0.0 < self.maximum_move_fraction <= 1.0:
            raise ValueError("move fraction must lie in (0, 1]")


@dataclass(frozen=True)
class LocalizationConsensus:
    offsets_um: np.ndarray
    selected: np.ndarray
    mean_magnitude_um: np.ndarray
    maximum_disagreement_um: np.ndarray
    maximum_sigma_um: np.ndarray
    maximum_safe_probability: np.ndarray
    minimum_direction_cosine: np.ndarray
    global_gate_passed: bool
    report: dict[str, Any]


def _member_direction_cosines(offsets: np.ndarray, mean: np.ndarray) -> np.ndarray:
    member_norm = np.linalg.norm(offsets, axis=2)
    mean_norm = np.linalg.norm(mean, axis=1)
    denominator = member_norm * mean_norm[None]
    numerator = np.sum(offsets * mean[None], axis=2)
    cosines = np.ones_like(numerator, dtype=np.float32)
    valid = denominator > 1e-6
    cosines[valid] = numerator[valid] / denominator[valid]
    # A member predicting no meaningful correction cannot support a nonzero vote.
    cosines[(member_norm <= 1e-6) & (mean_norm[None] > 1e-6)] = -1.0
    return cosines.min(axis=0)


def select_consensus_offsets(
    member_offsets_um: np.ndarray,
    member_sigma_um: np.ndarray,
    member_safe_probability: np.ndarray,
    *,
    policy: LocalizationConsensusPolicy = LocalizationConsensusPolicy(),
) -> LocalizationConsensus:
    offsets = np.asarray(member_offsets_um, dtype=np.float32)
    sigma = np.asarray(member_sigma_um, dtype=np.float32)
    safe = np.asarray(member_safe_probability, dtype=np.float32)
    if offsets.ndim != 3 or offsets.shape[2] != 3:
        raise ValueError("member offsets must have shape (M, N, 3)")
    if sigma.shape != offsets.shape or safe.shape != offsets.shape[:2]:
        raise ValueError("localization uncertainty inventories do not align")
    if len(offsets) < policy.minimum_members:
        raise ValueError("too few independently accepted localization members")
    if not np.isfinite(offsets).all() or not np.isfinite(sigma).all() or not np.isfinite(safe).all():
        raise ValueError("localization ensemble values must be finite")
    if np.any(sigma <= 0) or np.any((safe < 0.0) | (safe > 1.0)):
        raise ValueError("localization uncertainty values are outside their domains")

    mean = offsets.mean(axis=0)
    magnitude = np.linalg.norm(mean, axis=1)
    disagreement = np.linalg.norm(offsets - mean[None], axis=2).max(axis=0)
    maximum_sigma = sigma.max(axis=(0, 2))
    maximum_safe = safe.max(axis=0)
    minimum_cosine = _member_direction_cosines(offsets, mean)
    selected = (
        (magnitude >= policy.minimum_correction_um)
        & (magnitude <= policy.maximum_correction_um)
        & (disagreement <= policy.maximum_member_disagreement_um)
        & (maximum_sigma <= policy.maximum_predicted_sigma_um)
        & (maximum_safe <= policy.maximum_safe_probability)
        & (minimum_cosine >= policy.minimum_direction_cosine)
    )
    selected_fraction = float(selected.mean()) if len(selected) else 0.0
    global_gate = selected_fraction <= policy.maximum_move_fraction
    if not global_gate:
        selected = np.zeros_like(selected)
    applied = np.zeros_like(mean)
    applied[selected] = mean[selected] * policy.blend
    report = {
        "members": len(offsets),
        "nodes": len(mean),
        "selected_nodes": int(selected.sum()),
        "selected_fraction_before_global_gate": selected_fraction,
        "global_gate_passed": global_gate,
        "policy": {
            key: getattr(policy, key)
            for key in policy.__dataclass_fields__
        },
        "topology_preserving": True,
        "node_count_preserving": True,
        "member_subset_search": False,
        "ensemble_weight_search": False,
    }
    return LocalizationConsensus(
        offsets_um=applied,
        selected=selected,
        mean_magnitude_um=magnitude,
        maximum_disagreement_um=disagreement,
        maximum_sigma_um=maximum_sigma,
        maximum_safe_probability=maximum_safe,
        minimum_direction_cosine=minimum_cosine,
        global_gate_passed=global_gate,
        report=report,
    )


def apply_physical_coordinate_offsets(
    control: GraphData,
    offsets_um_by_node_id: dict[int, np.ndarray],
    *,
    voxel_size_zyx_um: np.ndarray = NATIVE_VOXEL_UM,
    spatial_shape_zyx: tuple[int, int, int] | None = None,
) -> GraphData:
    """Apply selected physical offsets while copying graph identity/topology."""

    from biohub_tracker.graphs import GraphData, GraphNode

    scale = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("voxel size must contain three positive finite values")
    shape = None if spatial_shape_zyx is None else np.asarray(spatial_shape_zyx, dtype=np.float32)
    if shape is not None and (shape.shape != (3,) or np.any(shape <= 0)):
        raise ValueError("spatial shape must contain three positive values")
    known_ids = {int(node.node_id) for node in control.nodes}
    if len(known_ids) != len(control.nodes):
        raise ValueError("control graph contains duplicate node identifiers")
    unknown = set(offsets_um_by_node_id) - known_ids
    if unknown:
        raise ValueError(f"coordinate offsets reference unknown nodes: {sorted(unknown)}")
    output: list[GraphNode] = []
    for node in control.nodes:
        node_id = int(node.node_id)
        if node_id not in offsets_um_by_node_id:
            output.append(node)
            continue
        physical = np.asarray(offsets_um_by_node_id[node_id], dtype=np.float32)
        if physical.shape != (3,) or not np.isfinite(physical).all():
            raise ValueError(f"node {node_id} offset must contain three finite values")
        if float(np.linalg.norm(physical)) > 10.0 + 1e-6:
            raise ValueError(f"node {node_id} offset exceeds the model support")
        coordinate = np.asarray((node.z, node.y, node.x), dtype=np.float32) + physical / scale
        if shape is not None and (np.any(coordinate < 0.0) or np.any(coordinate >= shape)):
            raise ValueError(f"node {node_id} refined coordinate leaves the image")
        output.append(
            GraphNode(node_id, int(node.t), float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))
        )
    refined = GraphData(nodes=tuple(output), edges=control.edges)
    if tuple((int(node.node_id), int(node.t)) for node in refined.nodes) != tuple(
        (int(node.node_id), int(node.t)) for node in control.nodes
    ) or refined.edges != control.edges:
        raise RuntimeError("coordinate refinement changed graph identity or topology")
    return refined
