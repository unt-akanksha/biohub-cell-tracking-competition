from copy import deepcopy
from pathlib import Path
import json
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/score-focus-division-preserving-assignment.py'))


def data():
    ref = json.loads((ROOT/'reports/experiments/focus-source-flow-v1-result.json').read_text())
    rows = dict(parent=ref['per_movie']['parent'], control=ref['per_movie']['candidate'], candidate=deepcopy(ref['per_movie']['candidate']))
    summaries = dict(parent=ref['summaries']['parent'], control=ref['summaries']['candidate'], candidate=deepcopy(ref['summaries']['candidate']))
    return rows,summaries,ref


def test_noop_fails_without_relaxing_original_regression_gate():
    result = M['comparison'](*data())
    assert not result['source_gate_passed']
    assert not result['conditions']['original_source_gate']
    assert not result['conditions']['score_gain_over_focus_flow']


def test_positive_average_does_not_override_movie_guard():
    rows,summaries,ref = data()
    summaries['candidate']['score'] += .01
    summaries['candidate']['edge_jaccard'] += .01
    rows['candidate'][0]['adj_edge_jaccard'] -= .03
    result = M['comparison'](rows,summaries,ref)
    assert not result['conditions']['per_movie_loss_bounded']
    assert not result['source_gate_passed']


def test_division_true_positive_loss_is_not_promoted():
    rows,summaries,ref = data()
    summaries['candidate']['division_tp'] -= 1
    assert not M['comparison'](rows,summaries,ref)['conditions']['true_divisions_preserved']


def test_changed_original_score_rejected():
    rows,summaries,ref = data()
    rows = deepcopy(rows)
    rows['control'][0]['edge_tp'] += 1
    with pytest.raises(ValueError,match='replay'):
        M['comparison'](rows,summaries,ref)


def test_actual_graph_roundtrip_preserves_coordinates_edges_and_empty_schema():
    for coords,edges in [(np.array([[0,1,2,3],[1,1,2,3]],np.float32),[(0,1)]),
                         (np.empty((0,4),np.float32),[])]:
        graph = M['graph_from_arrays'](coords,edges)
        assert graph.num_nodes() == len(coords)
        assert graph.num_edges() == len(edges)
        assert all(axis in graph.node_attr_keys() for axis in ('z','y','x'))
        if len(coords):
            M['SOURCE']['verify_graph'](graph,coords,[(0,1,0.)])
