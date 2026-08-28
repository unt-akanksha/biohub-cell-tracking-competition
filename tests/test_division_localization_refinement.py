from __future__ import annotations

import math

import pytest

from biohub_tracker.graphs import GraphData, GraphNode
from research.division_localization_refinement import (
    DivisionLocalizationError,
    apply_division_coordinate_donor,
    apply_division_localization,
    division_coordinate_donor_scope,
    division_localization_scope,
)


def graph(*, offset: float = 0.0, edges: tuple[tuple[int, int], ...] | None = None) -> GraphData:
    nodes = (
        GraphNode(10, 0, 1.0 + offset, 1.0, 1.0),
        GraphNode(11, 1, 2.0 + offset, 2.0, 2.0),
        GraphNode(12, 2, 3.0 + offset, 3.0, 3.0),
        GraphNode(13, 2, 4.0 + offset, 4.0, 4.0),
        GraphNode(14, 3, 5.0 + offset, 5.0, 5.0),
        GraphNode(20, 0, 6.0 + offset, 6.0, 6.0),
    )
    return GraphData(
        nodes=nodes,
        edges=edges if edges is not None else ((10, 11), (11, 12), (11, 13), (12, 14)),
    )


def test_scope_selects_only_division_parent_and_immediate_daughters() -> None:
    scope = division_localization_scope(graph())

    assert scope.division_parent_ids == (11,)
    assert scope.division_daughter_ids == (12, 13)
    assert scope.selected_node_ids == (11, 12, 13)


def test_application_preserves_graph_and_changes_only_selected_coordinates() -> None:
    control = graph()
    refined = graph(offset=0.25)
    output = apply_division_localization(control, refined)
    by_id = {int(node.node_id): node for node in output.nodes}
    base = {int(node.node_id): node for node in control.nodes}

    assert output.edges is control.edges
    assert tuple((node.node_id, node.t) for node in output.nodes) == tuple(
        (node.node_id, node.t) for node in control.nodes
    )
    for node_id in (11, 12, 13):
        assert by_id[node_id].z == pytest.approx(base[node_id].z + 0.25)
    for node_id in (10, 14, 20):
        assert by_id[node_id] is base[node_id]


def test_application_aligns_refined_nodes_by_identifier() -> None:
    control = graph()
    source = graph(offset=0.5)
    reversed_refined = GraphData(nodes=tuple(reversed(source.nodes)), edges=source.edges)

    output = apply_division_localization(control, reversed_refined)

    assert {node.node_id: node.z for node in output.nodes}[11] == pytest.approx(2.5)


def test_coordinate_donor_may_have_different_topology_and_extra_nodes() -> None:
    control = graph()
    source = graph(offset=0.75, edges=((10, 11), (11, 12)))
    donor = GraphData(
        nodes=source.nodes + (GraphNode(99, 8, 1.0, 1.0, 1.0),),
        edges=source.edges,
    )

    output = apply_division_coordinate_donor(control, donor)
    by_id = {int(node.node_id): node for node in output.nodes}

    assert by_id[11].z == pytest.approx(2.75)
    assert by_id[12].z == pytest.approx(3.75)
    assert by_id[13].z == pytest.approx(4.75)
    assert by_id[10].z == pytest.approx(1.0)
    assert output.edges is control.edges


def test_coordinate_donor_skips_an_incomplete_division_event() -> None:
    source = graph(offset=0.5)
    donor = GraphData(
        nodes=tuple(node for node in source.nodes if node.node_id != 13),
        edges=tuple(edge for edge in source.edges if 13 not in edge),
    )

    scope = division_coordinate_donor_scope(graph(), donor)
    output = apply_division_coordinate_donor(graph(), donor)

    assert scope.eligible_division_parent_ids == ()
    assert scope.eligible_node_ids == ()
    assert scope.unsupported_division_parent_ids == (11,)
    assert scope.missing_donor_node_ids == (13,)
    assert output == graph()


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (graph(edges=((10, 11), (11, 12))), "topology differs"),
        (
            GraphData(
                nodes=graph().nodes + (GraphNode(10, 0, 1.0, 1.0, 1.0),),
                edges=graph().edges,
            ),
            "duplicate node",
        ),
        (
            GraphData(
                nodes=tuple(
                    GraphNode(node.node_id, 4 if node.node_id == 13 else node.t, node.z, node.y, node.x)
                    for node in graph().nodes
                ),
                edges=graph().edges,
            ),
            "not consecutive",
        ),
        (
            GraphData(
                nodes=tuple(
                    GraphNode(node.node_id, node.t, math.nan if node.node_id == 11 else node.z, node.y, node.x)
                    for node in graph().nodes
                ),
                edges=graph().edges,
            ),
            "must be finite",
        ),
    ],
)
def test_application_fails_closed_on_malformed_refinement(
    candidate: GraphData, message: str
) -> None:
    with pytest.raises(DivisionLocalizationError, match=message):
        apply_division_localization(graph(), candidate)


def test_validator_rejects_merges_and_fake_forks() -> None:
    with pytest.raises(DivisionLocalizationError, match="merge"):
        division_localization_scope(
            graph(edges=((10, 11), (20, 11), (11, 12), (11, 13)))
        )
    fork_graph = GraphData(
        nodes=graph().nodes
        + (
            GraphNode(21, 1, 1.0, 1.0, 1.0),
            GraphNode(22, 1, 1.0, 1.0, 1.0),
        ),
        edges=((10, 11), (10, 21), (10, 22)),
    )
    with pytest.raises(DivisionLocalizationError, match="outdegree"):
        division_localization_scope(fork_graph)
