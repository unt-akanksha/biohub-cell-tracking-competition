import copy
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/summarize-calibrated-motion-selection.py'))


def fixture():
    values=runpy.run_path(str(ROOT/'tests/test_image_motion_linker_comparison.py'))['fixture']()
    candidate,manifest=values[:2]
    uncalibrated=copy.deepcopy(candidate)
    uncalibrated['summary']['score']-=.01
    calibration=dict(status='verified_calibration_not_tracking_candidate',profile='full',parameters=[.4,.3,-2.8],
        checkpoint_sha256=candidate['checkpoint_sha256'],source_sha256=dict(result='d'*64),
        result=dict(frozen_after=dict(neural='c'*64,flow='f'*64)))
    manifest['association_calibration']=dict(parameters=calibration['parameters'],result_sha256='d'*64,
        checkpoint_sha256=candidate['checkpoint_sha256'],frozen_hashes=calibration['result']['frozen_after'])
    return (*values,uncalibrated,calibration)


def test_calibrated_gain_over_neural_not_enough_if_flow_stronger():
    report=M['compare'](*fixture())
    assert report['delta_vs_uncalibrated']>0 and report['delta_vs_flow']<0
    assert report['decision']=='not_strongest_standalone' and not report['authorized_for_submission']


def test_changed_calibration_or_control_rejected():
    values=fixture(); values[1]['association_calibration']['result_sha256']='e'*64
    with pytest.raises(ValueError): M['compare'](*values)
    values=fixture(); values[7]['per_movie'][0]['num_pred_nodes']+=1
    with pytest.raises(ValueError): M['compare'](*values)
