import copy
import hashlib
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZE = runpy.run_path(str(ROOT/'scripts/summarize-causal-motion-selection.py'))['summarize']
COMPARE_STATIC = runpy.run_path(str(ROOT/'scripts/summarize-causal-motion-selection.py'))['compare_static']


def fixture():
    row = dict(stem='a', num_pred_nodes=4, node_recall=.9, total_node_ratio=.2, adj_edge_jaccard=.6)
    control = dict(checkpoint_sha256='a'*64, per_movie=[row], summary=dict(score=.6))
    result = dict(checkpoint_sha256='a'*64, target_audit_opened=False, authorized_for_submission=False,
                  fit_receipt_sha256=hashlib.sha256(b'fit').hexdigest(),
                  per_movie=dict(original=[copy.deepcopy(row)], motion=[copy.deepcopy(row)]),
                  summaries=dict(motion=dict(score=.6)), score_delta=0,
                  receipts=dict(a=dict(nodes_unchanged=True)))
    return result, control


def test_equal_candidate_is_rejected_not_promoted():
    result, control = fixture()
    assert SUMMARIZE(result, control, b'fit')['decision'] == 'reject'


def test_source_embedding_normalizes_only_crlf_not_content():
    result, control = fixture()
    result['fit_receipt_sha256'] = hashlib.sha256(b'fit\n').hexdigest()
    assert SUMMARIZE(result, control, b'fit\r\n')['decision'] == 'reject'
    with pytest.raises(ValueError):
        SUMMARIZE(result, control, b'different\r\n')


def test_static_control_requires_same_native_and_same_detection_metrics():
    result, _ = fixture()
    static = copy.deepcopy(result)
    assert COMPARE_STATIC(result, static)['causal_minus_static'] == 0
    static['per_movie']['motion'][0]['num_pred_nodes'] += 1
    with pytest.raises(ValueError):
        COMPARE_STATIC(result, static)
    static = copy.deepcopy(result)
    static['per_movie']['original'][0]['adj_edge_jaccard'] += .1
    with pytest.raises(ValueError):
        COMPARE_STATIC(result, static)


@pytest.mark.parametrize('mutation', ['nodes', 'native_score', 'coverage', 'fit', 'scope', 'delta'])
def test_inconsistent_evidence_rejected(mutation):
    result, control = fixture()
    if mutation == 'nodes':
        result['per_movie']['motion'][0]['num_pred_nodes'] += 1
    elif mutation == 'native_score':
        result['per_movie']['original'][0]['adj_edge_jaccard'] += .1
    elif mutation == 'coverage':
        result['per_movie']['motion'] = []
    elif mutation == 'fit':
        result['fit_receipt_sha256'] = '0'*64
    elif mutation == 'scope':
        result['target_audit_opened'] = True
    else:
        result['score_delta'] = .1
    with pytest.raises(ValueError):
        SUMMARIZE(result, control, b'fit')
