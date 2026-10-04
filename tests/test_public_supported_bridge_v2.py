import runpy
from pathlib import Path
import pytest
from research.public_supported_bridge_v2 import bridge

OLD = runpy.run_path(str(Path(__file__).with_name('test_public_supported_bridge.py')))


@pytest.mark.parametrize('name', [n for n in OLD if n.startswith('test_')])
def test_original_contract(name):
    function = OLD[name]
    original = function.__globals__['bridge']
    function.__globals__['bridge'] = lambda g, c, e: bridge(g, c, e, range(len(c)))
    try:
        if name == 'test_real_complete_bridge_preserves_parent':
            function(1); function(2)
        else:
            function()
    finally:
        function.__globals__['bridge'] = original


def test_synthetic_raw_id_collision_is_not_an_endpoint():
    graph, coords, edges = OLD['example']()
    graph['nodes']['1'] = dict(node_id=1, t=50, z=40, y=50, x=60)
    result, receipt = bridge(graph, coords, edges, {0, 2})
    assert result == graph and receipt['added_nodes'] == 0
