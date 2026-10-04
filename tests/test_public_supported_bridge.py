import copy
import numpy as np
import pytest
from research.public_supported_bridge import bridge


def example(missing=1):
    coords = np.asarray([[i, 20, 30, 40] for i in range(missing + 2)])
    nodes = {str(i): dict(node_id=i, t=int(coords[i, 0]), z=20, y=30, x=40)
             for i in (0, missing + 1)}
    # Fillers are far away; they exercise the fixed production node cap.
    for i in range(100, 200):
        nodes[str(i)] = dict(node_id=i, t=i % 100, z=50, y=200, x=200)
    graph = dict(nodes=nodes, edges=[])
    edges = np.asarray([[i, i + 1, .95, 0.] for i in range(missing + 1)])
    return graph, coords, edges


@pytest.mark.parametrize('missing', [1, 2])
def test_real_complete_bridge_preserves_parent(missing):
    graph, coords, edges = example(missing)
    if missing == 2:
        graph['nodes'].update({str(i): dict(node_id=i, t=i % 100, z=50, y=200, x=200)
                               for i in range(200, 300)})
    before = copy.deepcopy(graph)
    candidate, receipt = bridge(graph, coords, edges)
    assert graph == before
    assert receipt['added_nodes'] == missing and receipt['added_edges'] == missing + 1
    for i, row in before['nodes'].items():
        assert candidate['nodes'][i] == row
    assert candidate['nodes']['1']['x'] == 40


def test_weak_edge_or_too_long_gap_rejected():
    graph, coords, edges = example()
    edges[0, 2] = .879
    assert bridge(graph, coords, edges)[1]['added_nodes'] == 0
    assert bridge(*example(3))[1]['added_nodes'] == 0


def test_existing_same_frame_detection_prevents_duplicate():
    graph, coords, edges = example()
    graph['nodes']['150'].update(t=1, z=20, y=30, x=40)
    assert bridge(graph, coords, edges)[1]['added_nodes'] == 0


def test_competing_equal_edge_abstains_and_no_division_is_changed():
    graph, coords, edges = example()
    coords = np.concatenate((coords, [[1, 20, 31, 40]]))
    edges = np.concatenate((edges, [[0, 3, .95, 0.]]))
    assert bridge(graph, coords, edges)[1]['added_nodes'] == 0


def test_budget_and_motion_bound():
    graph, coords, edges = example()
    small = dict(nodes={k: v for k, v in graph['nodes'].items() if k in ('0', '2')}, edges=[])
    assert bridge(small, coords, edges)[1]['added_nodes'] == 0
    coords[1, 1] += 5
    assert bridge(graph, coords, edges)[1]['added_nodes'] == 0


def test_malformed_edge_rejected():
    graph, coords, edges = example()
    edges[0, 1] = 999
    with pytest.raises(ValueError, match='Dangling'):
        bridge(graph, coords, edges)
