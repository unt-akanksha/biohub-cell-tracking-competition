from copy import deepcopy
import pytest
from research.public_owned_flow_comparison import compare


def fixture():
    base = [dict(stem=s, num_pred_nodes=100, node_recall=.95,
                 edge_tp=9, edge_fp=1, edge_fn=1, adj_edge_jaccard=.8) for s in ('a', 'b')]
    rows = {a: deepcopy(base) for a in ('public_control', 'prior_consensus', 'candidate')}
    rows['candidate'][0].update(edge_tp=10, edge_fn=0, adj_edge_jaccard=.9)
    scores = {a: dict(score=.8, edge_jaccard=.8) for a in rows}
    scores['candidate'] = dict(score=.85, edge_jaccard=.85)
    return rows, scores


def test_requires_gain_over_both_controls():
    rows, scores = fixture()
    assert compare(rows, scores)['diagnostic_gate_passed']
    scores['prior_consensus'] = scores['candidate'].copy()
    assert not compare(rows, scores)['diagnostic_gate_passed']


def test_added_false_edge_rejected():
    rows, scores = fixture()
    rows['candidate'][0]['edge_fp'] += 1
    assert not compare(rows, scores)['diagnostic_gate_passed']


def test_single_movie_regression_rejected():
    rows, scores = fixture()
    rows['candidate'][1]['adj_edge_jaccard'] -= .001
    assert not compare(rows, scores)['diagnostic_gate_passed']


@pytest.mark.parametrize('field,value', [('num_pred_nodes', 99), ('node_recall', .94)])
def test_detection_changes_rejected(field, value):
    rows, scores = fixture()
    rows['candidate'][0][field] = value
    with pytest.raises(ValueError, match='preservation'):
        compare(rows, scores)


def test_nonfinite_and_mismatched_scope_rejected():
    rows, scores = fixture()
    scores['candidate']['score'] = float('nan')
    with pytest.raises(ValueError, match='Finite'):
        compare(rows, scores)
    rows['candidate'].reverse()
    with pytest.raises(ValueError, match='scope'):
        compare(rows, scores)


def test_diagnostic_pass_never_authorizes_submission():
    assert compare(*fixture())['authorized_for_submission'] is False
