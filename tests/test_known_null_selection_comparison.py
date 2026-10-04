import copy
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARE = runpy.run_path(str(ROOT/'scripts/summarize-known-null-selection.py'))['compare']


def inputs():
    rows = [dict(stem=f'm{i}',num_pred_nodes=30,node_recall=.95,total_node_ratio=.1,
        edge_tp=9,edge_fp=1,edge_fn=1,division_tp=0,division_fp=0,division_fn=1,
        adj_edge_jaccard=.8) for i in range(8)]
    candidate = dict(status='scored_complete_selection',checkpoint_sha256='a'*64,
        per_movie=rows,summary=dict(score=.8),target_audit_opened=False,authorized_for_submission=False)
    manifest = dict(status='completed',checkpoint_sha256='a'*64,
        known_null_training=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False),
        node_reference_manifest_sha256='r'*64,
        records=[dict(stem=r['stem'],reference_nodes_identical=True) for r in rows])
    training = dict(checkpoint_sha256='a'*64,steps=1000,detector_unchanged=True,probe_inputs_replayed=True)
    native = dict(checkpoint_sha256='c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470',
                  per_movie=copy.deepcopy(rows),summary=dict(score=.7))
    causal = dict(per_movie=dict(motion=copy.deepcopy(rows)),summaries=dict(motion=dict(score=.9)))
    return candidate,manifest,training,native,causal


def test_reports_both_controls_without_promoting_weak_improvement():
    result = COMPARE(*inputs())
    assert result['delta_vs_native'] > 0 and result['delta_vs_causal'] < 0
    assert result['authorized_for_submission'] is False


@pytest.mark.parametrize('mutation',['checkpoint','probe','nodes','coverage','scope','coordinate_receipt'])
def test_inconsistent_selection_evidence_is_rejected(mutation):
    candidate,manifest,training,native,causal = inputs()
    if mutation == 'checkpoint':
        candidate['checkpoint_sha256'] = 'b'*64
    elif mutation == 'probe':
        training['steps'] = 100
    elif mutation == 'nodes':
        candidate['per_movie'][0]['num_pred_nodes'] += 1
    elif mutation == 'coverage':
        candidate['per_movie'].pop()
    elif mutation == 'scope':
        candidate['target_audit_opened'] = True
    else:
        manifest['records'][0]['reference_nodes_identical'] = False
    with pytest.raises(ValueError):
        COMPARE(candidate,manifest,training,native,causal)
