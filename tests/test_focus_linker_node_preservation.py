from types import SimpleNamespace

import pytest

td = pytest.importorskip('tracksdata')
pl = pytest.importorskip('polars')
from research.focus_linker_runtime import solve_links_keep_nodes


@pytest.mark.parametrize('selected', [True, False])
def test_solver_cannot_remove_isolated_or_unselected_nodes(selected):
    graph = td.graph.InMemoryGraph()
    for axis in ('z', 'y', 'x'):
        graph.add_node_attr_key(axis, pl.Float64, 0.)
    ids = graph.bulk_add_nodes([{'t': 0, 'x': 2.5}, {'t': 1, 'x': 3.5}, {'t': 0, 'x': 99.}])
    graph.add_edge_attr_key('edge_prob', pl.Float64, 0.)
    graph.bulk_add_edges([dict(source_id=ids[0], target_id=ids[1], edge_prob=.9)])
    graph.add_edge_attr_key('chosen', pl.Boolean, selected)
    solver = SimpleNamespace(output_key='chosen', solve=lambda g: None)
    result = solve_links_keep_nodes(graph, solver)
    assert result.num_nodes() == 3
    assert result.num_edges() == int(selected)
    assert sorted(result.node_attrs()['x'].to_list()) == [2.5, 3.5, 99.]
    if selected:
        edge = result.edge_attrs().row(0, named=True)
        nodes = {r['node_id']: r for r in result.node_attrs().iter_rows(named=True)}
        assert nodes[edge['source_id']]['x'] == 2.5
        assert nodes[edge['target_id']]['x'] == 3.5
        assert edge['edge_prob'] == .9
