import copy
from pathlib import Path
import runpy

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/summarize-edge-feature-tta-score.py'))


def inputs():
    row = dict(stem='movie',num_pred_nodes=100,node_recall=.9,total_node_ratio=.1,
               edge_tp=7,edge_fp=2,edge_fn=3,division_tp=1,division_fp=1,division_fn=1,
               edge_jaccard=.5,adj_edge_jaccard=.495)
    scored = dict(status='scored_complete_selection',checkpoint_sha256=MODULE['SOURCE_SHA'],
        target_audit_opened=False,authorized_for_submission=False,per_movie=[row],summary=dict(score=.5))
    manifest = dict(status='completed',checkpoint_sha256=MODULE['SOURCE_SHA'],
        records=[dict(stem='movie',reference_nodes_identical=True)],
        edge_feature_tta=dict(views=8,maximum_mean_absolute_feature_delta=.1),
        node_reference_manifest_sha256='a'*64)
    return copy.deepcopy(scored),scored,manifest,['movie']


def test_valid_equal_detections_and_changed_links():
    candidate,control,manifest,movies = inputs()
    candidate['per_movie'][0]['edge_tp'] += 1
    candidate['summary']['score'] += .01
    result = MODULE['summarize'](candidate,control,manifest,movies)
    assert result['detection_metrics_identical'] and result['score_delta'] == pytest.approx(.01)
    assert result['per_movie_deltas'][0]['edge_tp'] == 1


@pytest.mark.parametrize('key,value',[('num_pred_nodes',99),('node_recall',.89),('total_node_ratio',.09)])
def test_detection_changes_rejected(key,value):
    candidate,control,manifest,movies = inputs()
    candidate['per_movie'][0][key] = value
    with pytest.raises(ValueError,match='detection evidence'):
        MODULE['summarize'](candidate,control,manifest,movies)


def test_missing_preservation_receipt_rejected():
    candidate,control,manifest,movies = inputs()
    manifest['records'][0]['reference_nodes_identical'] = False
    with pytest.raises(ValueError,match='preservation'):
        MODULE['summarize'](candidate,control,manifest,movies)
