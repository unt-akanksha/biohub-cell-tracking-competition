from pathlib import Path
import runpy
import pytest

td = pytest.importorskip('tracksdata')
from research.empty_graph_schema import restore_empty_spatial_schema


def test_empty_geff_roundtrip_can_be_scored_without_adding_nodes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    replay = runpy.run_path(str(root/'scripts/replay-focus-bridge-official.py'))
    graph = replay['prediction_graph']({'nodes':{}, 'edges':[]})
    graph.to_geff(tmp_path/'empty.geff')
    restored = td.graph.IndexedRXGraph.from_geff(str(tmp_path/'empty.geff'))[0]
    restore_empty_spatial_schema(restored)
    assert restored.num_nodes() == restored.num_edges() == 0
    assert all(k in restored.copy().node_attr_keys() for k in ('z','y','x'))
    truth = replay['prediction_graph']({'nodes':{
        '0':dict(t=0,z=1.,y=1.,x=1.), '1':dict(t=1,z=1.,y=1.,x=1.)},
        'edges':[dict(source_id=0,target_id=1)]})
    result = replay['load_scorer']('current').evaluate(restored, truth,
        scale=(1.625,.40625,.40625), max_distance=7.)
    assert result.edge_tp == result.edge_fp == result.num_pred_nodes == 0
    assert result.edge_fn == 1


def test_never_invents_coordinates_for_nonempty_graph():
    graph = td.graph.InMemoryGraph()
    graph.bulk_add_nodes([dict(t=0)])
    with pytest.raises(ValueError, match='Nonempty'):
        restore_empty_spatial_schema(graph)
