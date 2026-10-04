import json
from pathlib import Path
import pytest
from research.detector_calibration_records import calibration_scope
from research.detector_confidence_calibration import receipt,SPARSE_SHA,FULL_NOTEBOOK_SHA


def fixture():
    root=Path(__file__).resolve().parents[1]
    split=json.loads((root/'research/independent_real_baseline_v1_split.json').read_text())
    groups=calibration_scope(split,False)
    frames=[]; movies=[]
    for group,stems in groups.items():
        for stem in stems:
            frames.extend(dict(stem=stem,group=group,t=t,annotations=10,parent_matched=9,candidate_calibrated_matched=9) for t in (0,1,2))
            movies.append(dict(stem=stem,group=group,annotations=30,parent_matched=27,candidate_calibrated_matched=27))
    report=dict(status='verified_training_calibration_collection_full',notebook_sha256=FULL_NOTEBOOK_SHA,
        full_calibration_completed=True,full_diagnostic_recall_preserved=True,selection_opened=False,
        target_audit_opened=False,authorized_for_submission=False,independent_validation=False,
        replayed_frames=18,maximum_replay_probability_delta=0.,per_frame=frames,per_movie=movies,
        collection=dict(candidate_sha256=SPARSE_SHA,groups=groups,models_unchanged=True),
        fit=dict(status='fitted_training_only_threshold',maximum_recall_loss=.005,threshold=.75,
            baseline_threshold=.5744425058364868,annotated_nodes=2880,parent_matched=2592,required_matched=2578))
    return report,split


@pytest.mark.parametrize('fault',[None,'diagnostic_flag','hidden_movie_loss','replay','checkpoint','scope','frames','threshold','fit'])
def test_only_full_training_diagnostic_pass_can_supply_cutoff(fault):
    report,split=fixture(); sha=SPARSE_SHA
    if fault=='diagnostic_flag': report['full_diagnostic_recall_preserved']=False
    if fault=='hidden_movie_loss':
        report['per_frame'][-1]['candidate_calibrated_matched']-=1
        report['per_movie'][-1]['candidate_calibrated_matched']-=1
    if fault=='replay': report['replayed_frames']=17
    if fault=='checkpoint': sha='f'*64
    if fault=='scope': report['target_audit_opened']=True
    if fault=='frames': report['per_frame'].pop()
    if fault=='threshold': report['fit']['threshold']=1.
    if fault=='fit': report['fit']['required_matched']-=1
    payload=json.dumps(report).encode()
    if fault:
        with pytest.raises(ValueError): receipt(payload,split,sha)
    else:
        result=receipt(payload,split,sha)
        assert result['threshold']==.75 and not result['authorized_for_submission']
