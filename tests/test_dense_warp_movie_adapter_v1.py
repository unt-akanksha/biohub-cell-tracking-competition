from pathlib import Path
import json
import pytest
from research.dense_warp_movie_adapter_v1 import build_runtime,compare_control

ROOT=Path(__file__).resolve().parents[1]


def test_runtime_preserves_original_views_and_three_fixed_arms():
    source=(ROOT/'scripts/run-trajectory-division-full-movie-v1.py').read_text()
    result=build_runtime(source)
    assert "for arm in ('original', 'warp44', 'warp6')" in result
    assert "filename = 'public-predictor-original.py'" in result
    assert "if arm == 'original' else post_source" not in result
    assert 'np.testing.assert_array_equal(coords, control_coords)' in result
    assert "secondary_link_mode='low_margin_consensus'" in result
    assert 'threshold=.48' in result


def test_reject_source_drift():
    with pytest.raises(ValueError):build_runtime('changed source')


def test_control_checks_graph_semantics_not_edge_order(tmp_path):
    graph=dict(nodes={'1':dict(t=0)},edges=[dict(source_id=1,target_id=2),dict(source_id=2,target_id=3)])
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    compare_control({**graph,'edges':graph['edges'][::-1]},path)
    with pytest.raises(ValueError):compare_control({**graph,'edges':graph['edges'][:1]},path)
