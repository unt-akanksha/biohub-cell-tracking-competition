import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/summarize-owned-detector-selection.py'))


def fixture():
    baseline = json.loads((ROOT/'reports/experiments/detector-spatial-tta-selection-v1-score.json').read_text())
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    pair = dict(status='verified_detector_fit_pair_not_selection',paired_inputs_identical=True,
                paired_targets_identical=True,authorized_for_submission=False,arms={})
    scores, manifests = {}, {}
    for arm, gain in [('sparse', .01), ('pu', .03)]:
        score = copy.deepcopy(baseline['result'])
        digest = ('a' if arm=='sparse' else 'b')*64
        score['checkpoint_sha256'] = digest
        for field in ('score','edge_jaccard'): score['summary'][field] += gain
        for row in score['per_movie']: row['adj_edge_jaccard'] += gain
        scores[arm] = score
        pair['arms'][arm] = dict(checkpoint_sha256=digest)
        manifests[arm] = dict(status='completed',checkpoint_sha256=digest,
            owned_detector_fit=dict(objective=arm,steps=1000,initialization_sha256=M['INITIAL_SHA']),
            detector_spatial_tta=dict(views=8,features='native unchanged',encode_calls=792),
            standalone_image_flow=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,flow_checkpoint_sha256=M['FLOW_SHA']),
            records=[dict(stem=s,processed_frames=100,image_shape=[100,64,256,256]) for s in split['folds'][0]['selection']],
            target_audit_opened=False,ground_truth_opened=False,authorized_for_submission=False)
    return scores, manifests, pair, baseline, split


def test_both_controls_and_worst_movies_reported_without_submission_claim():
    report = M['compare'](*fixture())
    assert report['chosen_arm_for_further_evaluation']=='pu'
    assert set(report['worst_movies'])=={'frozen','sparse','pu'}
    assert not report['authorized_for_submission'] and not report['target_audit_opened']


@pytest.mark.parametrize('fault',['recall','raw_edge','one_movie','worst_movie','majority','control_better'])
def test_pu_not_promoted_on_aggregate_score_alone(fault):
    scores, manifests, pair, baseline, split = fixture()
    pu = scores['pu']
    if fault=='recall': pu['summary']['node_recall'] -= .02
    if fault=='raw_edge': pu['summary']['edge_jaccard'] -= .1
    if fault=='one_movie': pu['per_movie'][0]['adj_edge_jaccard'] -= .1
    if fault=='worst_movie': min(pu['per_movie'],key=lambda r:r['adj_edge_jaccard'])['adj_edge_jaccard'] -= .1
    if fault=='majority':
        for row in pu['per_movie'][:5]: row['adj_edge_jaccard'] -= .04
    if fault=='control_better': scores['sparse']['summary']['score'] += .1
    report = M['compare'](scores,manifests,pair,baseline,split)
    assert report['chosen_arm_for_further_evaluation']!='pu'


@pytest.mark.parametrize('fault',['partial','order','checkpoint','policy','scope','nonfinite'])
def test_incomplete_or_incomparable_predictions_rejected(fault):
    args = fixture()
    scores, manifests, _, _, _ = args
    if fault=='partial': manifests['pu']['records'][0]['processed_frames']=99
    if fault=='order': scores['pu']['per_movie'].reverse()
    if fault=='checkpoint': scores['pu']['checkpoint_sha256']='f'*64
    if fault=='policy': manifests['pu']['standalone_image_flow']['null_logit']=-8
    if fault=='scope': scores['pu']['target_audit_opened']=True
    if fault=='nonfinite': scores['pu']['summary']['score']=float('nan')
    with pytest.raises(ValueError): M['compare'](*args)
