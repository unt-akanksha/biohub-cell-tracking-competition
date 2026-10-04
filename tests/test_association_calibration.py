from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from association_calibration import pack_sample,combine,objective,fit,metrics,FLOW_CONTROL


def example():
    return pack_sample([[0.,2.,100.],[2.,0.,-100.]],[[0.,-2.,99.],[-2.,0.,-99.]],
        [[1,0,0],[0,0,0]],[False,True,False])


def test_unknowns_ignored_and_null_real_target():
    data = example()
    assert data['x'].shape == (6,3)
    assert data['labels'].tolist() == [0,5]
    assert metrics(FLOW_CONTROL,data)['known_null_columns'] == 1
    assert np.max(np.abs(data['x'])) == 2


def test_gradient_and_group_softmax():
    data = combine([example(),example()]); theta = [.3,1.1,-2.]
    loss,gradient = objective(theta,data)
    for axis in range(3):
        delta = np.eye(3)[axis]*1e-5
        numerical = (objective(np.asarray(theta)+delta,data)[0]-objective(np.asarray(theta)-delta,data)[0])/2e-5
        assert gradient[axis] == pytest.approx(numerical,abs=1e-7)
    p = objective(theta,data,probabilities=True)
    assert np.add.reduceat(p,data['starts']) == pytest.approx(np.ones(4))
    assert np.isfinite(loss)


def test_fit_is_bounded_and_does_not_worsen_training_control():
    data = combine([example()]*5); result = fit(data)
    assert result['converged']
    assert result['fit_metrics']['nll'] <= result['flow_control']['nll']
    for value,(low,high) in zip(result['parameters'],result['bounds']):
        assert low <= value <= high


def test_bad_supervision_rejected():
    with pytest.raises(ValueError):
        pack_sample([[0]],[[0]],[[1]],[True])
    with pytest.raises(ValueError):
        combine([pack_sample([[0]],[[0]],[[0]],[False])])
    with pytest.raises(ValueError):
        pack_sample([[float('nan')]],[[0]],[[1]],[False])


def test_no_source_detections_still_represents_known_null():
    data = pack_sample(np.empty((0,1)),np.empty((0,1)),np.empty((0,1)),[True])
    assert metrics(FLOW_CONTROL,data)['correct'] == 1
    assert objective(FLOW_CONTROL,data)[0] == 0
