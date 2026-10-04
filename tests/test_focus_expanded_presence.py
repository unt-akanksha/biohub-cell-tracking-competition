import json
from pathlib import Path
import runpy
import numpy as np
import pytest
from research.focus_summary_validation import validate_summary

ROOT=Path(__file__).resolve().parents[1]
MODULE=runpy.run_path(str(ROOT/'scripts/fit-focus-expanded-presence.py'))


def original_fitting():
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-presence-summary-v1/focus_presence_summary/outputs'
    result=json.loads((folder/'result.json').read_text())
    record=next(r for r in result['records'] if r['role']=='fitting')
    with np.load(folder/(record['stem']+'.npz'),allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    labels=json.loads((ROOT/'.biohub/cache/focus-adaptation-labels-v1'/(record['stem']+'.json')).read_text())
    return arrays,record,labels


def test_actual_existing_fitting_summary_matches_full_label_audit():
    arrays,record,labels=original_fitting()
    replay=validate_summary(arrays,record,labels)
    assert replay['known_parent']==492 and replay['known_absent']==20


@pytest.mark.parametrize('change',['target','label','missing','nan','frame','dtype'])
def test_summary_corruption_rejected(change):
    arrays,record,labels=original_fitting()
    if change=='target':arrays['target_indices'][0]+=1
    elif change=='label':arrays['present'][0]=1-arrays['present'][0]
    elif change=='missing':arrays.pop('offset')
    elif change=='nan':arrays['context'][0,0]=np.nan
    elif change=='frame':arrays['source_frame'][0]+=1
    elif change=='dtype':arrays['target_indices']=arrays['target_indices'].astype(float)
    with pytest.raises(ValueError):validate_summary(arrays,record,labels)


def test_fit_export_precedes_corrected_diagnostic_and_no_overwrite(tmp_path,monkeypatch):
    function=MODULE['execute_fit'];space=function.__globals__;fit_path=tmp_path/'fit.json'
    fitting=object();diagnostic=object();events=[]
    baseline=dict(nll=1.,known_parent=10,known_absent=3,correct_parent=8,correct_absent=2)
    evidence=dict(worker_result_sha256='synthetic',unchanged_diagnostic_summaries={},baseline=baseline,physical=baseline)
    def fitting_only(data,role):
        assert data is fitting and role=='fitting' and not fit_path.exists()
        events.append('fit');return dict(theta=[1.],role=role)
    def checked_metrics(data,model):
        assert fit_path.exists() and json.loads(fit_path.read_text())['model']==model
        events.append('diagnostic' if data is diagnostic else 'fitting_metrics')
        return dict(baseline,nll=.5)
    monkeypatch.setitem(space,'fit',fitting_only);monkeypatch.setitem(space,'metrics',checked_metrics)
    _,_,gate=function(dict(fitting=fitting,diagnostic=diagnostic),evidence,fit_path)
    assert events==['fit','fitting_metrics','diagnostic'] and gate['passed']
    before=fit_path.read_bytes()
    with pytest.raises(ValueError,match='overwrite'):
        function(dict(fitting=fitting,diagnostic=diagnostic),evidence,fit_path)
    assert fit_path.read_bytes()==before
