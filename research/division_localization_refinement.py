"""Topology-preserving coordinate refinement around predicted divisions.

This module deliberately contains no learned model, metric, distance threshold,
or promotion rule. It only defines the small set of nodes a separately trained
coordinate refiner may change and fails closed if the two graph artifacts are
not exactly aligned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from biohub_tracker.graphs import GraphData, GraphNode


class DivisionLocalizationError(ValueError):
    """Raised when a coordinate-only refinement would be unsafe."""


@dataclass(frozen=True)
class DivisionLocalizationScope:
    division_parent_ids: tuple[int, ...]
    division_daughter_ids: tuple[int, ...]
    selected_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class DivisionCoordinateDonorScope:
    eligible_division_parent_ids: tuple[int, ...]
    eligible_node_ids: tuple[int, ...]
    unsupported_division_parent_ids: tuple[int, ...]
    missing_donor_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class _ValidatedGraph:
    nodes: dict[int, GraphNode]
    edges: frozenset[tuple[int, int]]
    outdegree: dict[int, int]
    daughters: dict[int, tuple[int, ...]]


def _integer(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not float(value).is_integer()
    ):
        raise DivisionLocalizationError(f"{label} must be a finite integer")
    return int(value)


def _finite_coordinate(value: Any, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DivisionLocalizationError(f"{label} must be finite")


def _validate(data: GraphData, role: str) -> _ValidatedGraph:
    nodes: dict[int, GraphNode] = {}
    for raw in data.nodes:
        node_id = _integer(raw.node_id, f"{role} node_id")
        time = _integer(raw.t, f"{role} node {node_id} time")
        if node_id < 0 or time < 0:
            raise DivisionLocalizationError(f"{role} node identifiers and times must be nonnegative")
        if node_id in nodes:
            raise DivisionLocalizationError(f"{role} contains duplicate node {node_id}")
        for axis, value in (("z", raw.z), ("y", raw.y), ("x", raw.x)):
            _finite_coordinate(value, f"{role} node {node_id} {axis}")
        nodes[node_id] = GraphNode(node_id, time, raw.z, raw.y, raw.x)

    edges: set[tuple[int, int]] = set()
    indegree = {node_id: 0 for node_id in nodes}
    outdegree = {node_id: 0 for node_id in nodes}
    daughters: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    for raw_source, raw_target in data.edges:
        source = _integer(raw_source, f"{role} edge source")
        target = _integer(raw_target, f"{role} edge target")
        edge = (source, target)
        if edge in edges:
            raise DivisionLocalizationError(f"{role} contains duplicate edge {source}->{target}")
        if source not in nodes or target not in nodes:
            raise DivisionLocalizationError(f"{role} contains dangling edge {source}->{target}")
        if source == target:
            raise DivisionLocalizationError(f"{role} contains self-loop {source}->{target}")
        if int(nodes[target].t) != int(nodes[source].t) + 1:
            raise DivisionLocalizationError(f"{role} edge {source}->{target} is not consecutive")
        edges.add(edge)
        indegree[target] += 1
        outdegree[source] += 1
        daughters[source].append(target)

    if any(value > 1 for value in indegree.values()):
        raise DivisionLocalizationError(f"{role} contains a merge")
    if any(value > 2 for value in outdegree.values()):
        raise DivisionLocalizationError(f"{role} contains an outdegree greater than two")
    return _ValidatedGraph(
        nodes=nodes,
        edges=frozenset(edges),
        outdegree=outdegree,
        daughters={key: tuple(sorted(value)) for key, value in daughters.items()},
    )


def division_localization_scope(control: GraphData) -> DivisionLocalizationScope:
    """Return the predicted division parents and immediate daughters."""

    validated = _validate(control, "control")
    parents = tuple(
        sorted(node_id for node_id, degree in validated.outdegree.items() if degree == 2)
    )
    daughters = tuple(
        sorted({daughter for parent in parents for daughter in validated.daughters[parent]})
    )
    return DivisionLocalizationScope(
        division_parent_ids=parents,
        division_daughter_ids=daughters,
        selected_node_ids=tuple(sorted(set(parents) | set(daughters))),
    )


def apply_division_localization(control: GraphData, refined: GraphData) -> GraphData:
    """Use refined coordinates only at predicted division parents/daughters.

    Node order, identifiers, times, and edges are copied from ``control``.
    Refined nodes may arrive in a different order, but must have identical node
    identifiers, times, and topology.
    """

    base = _validate(control, "control")
    candidate = _validate(refined, "refined")
    if base.nodes.keys() != candidate.nodes.keys():
        raise DivisionLocalizationError("control and refined node identifiers differ")
    if base.edges != candidate.edges:
        raise DivisionLocalizationError("control and refined topology differs")
    for node_id in base.nodes:
        if int(base.nodes[node_id].t) != int(candidate.nodes[node_id].t):
            raise DivisionLocalizationError(f"control and refined times differ at node {node_id}")

    parents = {
        node_id for node_id, degree in base.outdegree.items() if degree == 2
    }
    selected = parents | {
        daughter for parent in parents for daughter in base.daughters[parent]
    }
    output: list[GraphNode] = []
    for raw in control.nodes:
        node_id = int(raw.node_id)
        if node_id in selected:
            replacement = candidate.nodes[node_id]
            output.append(
                GraphNode(node_id, int(raw.t), replacement.z, replacement.y, replacement.x)
            )
        else:
            output.append(raw)
    return GraphData(nodes=tuple(output), edges=control.edges)


def apply_division_coordinate_donor(control: GraphData, donor: GraphData) -> GraphData:
    """Use a raw graph as a coordinate donor for a processed control graph.

    Production post-processing may remove nodes and add division edges after
    coordinate inference, so the donor topology is not expected to equal the
    processed topology. The donor must contain every selected processed node at
    the same time. Its edges and all non-selected coordinates are ignored.
    """

    base = _validate(control, "control")
    source = _validate(donor, "donor")
    scope = _division_coordinate_donor_scope(base, source)
    selected = set(scope.eligible_node_ids)

    output: list[GraphNode] = []
    for raw in control.nodes:
        node_id = int(raw.node_id)
        if node_id in selected:
            replacement = source.nodes[node_id]
            output.append(
                GraphNode(node_id, int(raw.t), replacement.z, replacement.y, replacement.x)
            )
        else:
            output.append(raw)
    return GraphData(nodes=tuple(output), edges=control.edges)


def _division_coordinate_donor_scope(
    base: _ValidatedGraph, source: _ValidatedGraph
) -> DivisionCoordinateDonorScope:
    eligible_parents: list[int] = []
    eligible_nodes: set[int] = set()
    unsupported_parents: list[int] = []
    missing_nodes: set[int] = set()
    parents = sorted(
        node_id for node_id, degree in base.outdegree.items() if degree == 2
    )
    for parent in parents:
        event_nodes = {parent, *base.daughters[parent]}
        event_missing = event_nodes - source.nodes.keys()
        if event_missing:
            unsupported_parents.append(parent)
            missing_nodes.update(event_missing)
            continue
        for node_id in event_nodes:
            if int(base.nodes[node_id].t) != int(source.nodes[node_id].t):
                raise DivisionLocalizationError(
                    f"control and donor times differ at selected node {node_id}"
                )
        eligible_parents.append(parent)
        eligible_nodes.update(event_nodes)
    return DivisionCoordinateDonorScope(
        eligible_division_parent_ids=tuple(eligible_parents),
        eligible_node_ids=tuple(sorted(eligible_nodes)),
        unsupported_division_parent_ids=tuple(unsupported_parents),
        missing_donor_node_ids=tuple(sorted(missing_nodes)),
    )


def division_coordinate_donor_scope(
    control: GraphData, donor: GraphData
) -> DivisionCoordinateDonorScope:
    """Report complete division events supported by a coordinate donor.

    An event is eligible only when its parent and both immediate daughters are
    all present. Incomplete events are preserved exactly instead of receiving a
    partial coordinate intervention.
    """

    return _division_coordinate_donor_scope(
        _validate(control, "control"), _validate(donor, "donor")
    )
