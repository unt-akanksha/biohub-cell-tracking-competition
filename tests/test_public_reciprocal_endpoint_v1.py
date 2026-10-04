import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

reconnect = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              'research/public_reciprocal_endpoint_v1.py'))['reconnect']


def fixture():
    coords = np.asarray([[t, 10, 30, 30 + t] for t in range(6)])
    nodes = {str(i): dict(node_id=i, t=int(t), z=int(z), y=int(y), x=int(x))
             for i, (t, z, y, x) in enumerate(coords)}
    graph = dict(nodes=nodes, edges=[dict(source_id=a, target_id=b)
                                    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5))])
    return graph, coords, np.asarray([[2, 3, .95, 1.]])


def test_reconnects_motion_supported_break_without_mutation():
    graph, coords, raw = fixture(); old = copy.deepcopy(graph)
    result, details = reconnect(graph, coords, raw, range(6))
    assert details['added_edges'] == 1 and details['added_nodes'] == 0
    assert result['nodes'] == graph['nodes'] and graph == old
    assert result['edges'][-1] == dict(source_id=2, target_id=3)


def test_blocks_synthetic_raw_id_reuse():
    graph, coords, raw = fixture()
    assert reconnect(graph, coords, raw, {0, 1, 2, 4, 5})[1]['added_edges'] == 0


def test_all_raw_alternatives_and_ties_participate():
    graph, coords, raw = fixture()
    coords = np.vstack([coords, [3, 10, 32, 33]])
    for p in (.95, .99):
        competing = np.vstack([raw, [2, 6, p, 1.]])
        assert reconnect(graph, coords, competing, range(6))[1]['added_edges'] == 0


def test_blocks_motion_disagreement_and_missing_history():
    graph, coords, raw = fixture()
    graph['nodes']['0']['x'] = 0
    assert reconnect(graph, coords, raw, range(6))[1]['added_edges'] == 0
    graph, coords, raw = fixture(); graph['edges'].pop(0)
    assert reconnect(graph, coords, raw, range(6))[1]['added_edges'] == 0


def test_existing_division_not_overwritten():
    graph, coords, raw = fixture()
    graph['edges'].append(dict(source_id=2, target_id=3))
    assert reconnect(graph, coords, raw, range(6))[1]['added_edges'] == 0


def test_invalid_raw_edge_rejected():
    graph, coords, raw = fixture(); raw[0, 1] = 4
    with pytest.raises(ValueError, match='adjacent'):
        reconnect(graph, coords, raw, range(6))
