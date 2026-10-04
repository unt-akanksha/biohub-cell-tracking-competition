import copy
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/summarize-detector-spatial-tta-selection.py'))


def fixture():
    stems=['6bba_'+str(i) for i in range(8)]
    rows=[dict(stem=s,edge_tp=10,edge_fp=2,edge_fn=2,division_tp=0,division_fp=0,division_fn=0,
        num_pred_nodes=100,node_recall=.95,adj_edge_jaccard=.7) for s in stems]
    flow=dict(flow_checkpoint_sha256=M['FLOW_SHA'],per_movie=dict(motion=copy.deepcopy(rows)),
        summaries=dict(motion=dict(score=.7,edge_jaccard=.75,node_recall=.95,division_jaccard=0.)))
    candidate=dict(status='scored_complete_selection',checkpoint_sha256=M['CHECKPOINT_SHA'],per_movie=rows,
        summary=dict(score=.72,edge_jaccard=.77,node_recall=.95,division_jaccard=0.),
        target_audit_opened=False,authorized_for_submission=False)
    manifest=dict(status='completed',checkpoint_sha256=M['CHECKPOINT_SHA'],
        standalone_image_flow=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,flow_checkpoint_sha256=M['FLOW_SHA']),
        detector_spatial_tta=dict(views=8,features='native unchanged',maximum_mean_absolute_logit_delta=.1,encode_calls=792),
        records=[dict(stem=s,processed_frames=100,image_shape=[100,64,256,256]) for s in stems],
        target_audit_opened=False,ground_truth_opened=False,authorized_for_submission=False)
    return candidate,manifest,flow


def test_positive_source_gain_still_needs_embryo_audit():
    report=M['compare'](*fixture())
    assert report['decision']=='source_selection_gain_requires_embryo_audit'
    assert not report['authorized_for_submission']


@pytest.mark.parametrize('fault',['recall','raw_jaccard'])
def test_score_gain_without_detection_or_raw_tracking_integrity_is_insufficient(fault):
    candidate,manifest,flow=fixture()
    candidate['summary']['node_recall' if fault=='recall' else 'edge_jaccard']=.5
    assert M['compare'](candidate,manifest,flow)['decision']=='no_robust_source_selection_gain'


def test_wrong_flow_partial_movies_and_changed_policy_fail_closed():
    for fault in ('flow','coverage','policy'):
        candidate,manifest,flow=fixture()
        if fault=='flow': flow['flow_checkpoint_sha256']='f'*64
        if fault=='coverage': manifest['records'][0]['processed_frames']=99
        if fault=='policy': manifest['standalone_image_flow']['null_logit']=-12.
        with pytest.raises(ValueError): M['compare'](candidate,manifest,flow)
