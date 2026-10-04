from pathlib import Path
import runpy
import copy
import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'research/trajectory_source_mixture_v1.py'))


def fake(final, coords, edges, ids, model, threshold):
    links = model['links']; graph = copy.deepcopy(final)
    return graph, dict(added_nodes=0, added_edges=len(links), links=links)


def test_union_deduplicates_and_retains_nodes():
    edge = dict(source_id=1, target_id=2)
    models = [dict(threshold=1, links=[edge]), dict(threshold=2, links=[edge])]
    final = dict(nodes={'1': {}}, edges=[])
    result, receipt = M['reconnect'](final, None, None, None, models, fake)
    assert result['edges'] == [edge] and result['nodes'] == final['nodes']
    assert receipt['links'][0]['supporting_experts'] == [0, 1]
    assert final['edges'] == []


def test_conflicting_experts_are_rejected():
    models = [dict(threshold=1, links=[dict(source_id=1, target_id=2)]),
              dict(threshold=1, links=[dict(source_id=1, target_id=3)])]
    with pytest.raises(ValueError, match='conflicting'):
        M['reconnect'](dict(nodes={}, edges=[]), None, None, None, models, fake)
