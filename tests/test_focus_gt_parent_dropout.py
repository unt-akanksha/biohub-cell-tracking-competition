import numpy as np
import pytest
from research.focus_gt_parent_dropout import augment


def sample():
    return dict(stem='fit',role='fitting',packet=dict(source_frame=np.array(0),source_indices=np.arange(3,dtype=np.int64),
        source_coords=np.array([[0.,0.,0.],[0.,0.,20.],[0.,0.,100.]],np.float32),source_features=np.zeros((3,32),np.float32),
        source_pos=np.zeros((3,32),np.float32),target_features=np.ones((4,32),np.float32),labels=np.array([0,0,3,-1],np.int64)))


def test_exact_center_drops_only_match_neighborhood():
    s=sample();out=augment(s,{0:[0.,0.,0.]})
    np.testing.assert_array_equal(out['packet']['source_indices'],[1,2])
    np.testing.assert_array_equal(out['packet']['labels'],[2,2,2,-1])
    np.testing.assert_array_equal(s['packet']['source_indices'],[0,1,2])
    np.testing.assert_array_equal(out['packet']['target_features'],s['packet']['target_features'])
    assert out['provenance']['synthetic_null_columns']==[0,1]


def test_gt_center_not_predicted_center_defines_absence():
    s=sample();out=augment(s,{0:[0.,0.,10.]})
    np.testing.assert_array_equal(out['packet']['source_indices'],[2])


def test_invalid_role_center_and_parent_inventory_rejected():
    with pytest.raises(ValueError):augment(dict(role='diagnostic'),{})
    with pytest.raises(ValueError):augment(sample(),{})
    with pytest.raises(ValueError):augment(sample(),{0:[20.,0.,0.]})


def test_exact_seven_micron_boundary_is_removed():
    s=sample();s['packet']['source_coords']=np.array([[0.,0.,0.],[7/1.625,0.,0.],[20.,0.,0.]],float)
    out=augment(s,{0:[0.,0.,0.]})
    np.testing.assert_array_equal(out['packet']['source_indices'],[2])
