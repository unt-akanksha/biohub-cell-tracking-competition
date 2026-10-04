import numpy as np
import pytest
from research.focus_joint_probe import select_samples,normalize_pair,ImagePairs


def sample(i,n,nulls):
    return dict(stem='fitting',role='fitting',packet=dict(source_frame=np.array(i),source_indices=np.arange(n),target_indices=np.arange(n),labels=np.array([n]*nulls+[0]*(n-nulls))))


def test_four_stress_pairs_include_largest_and_null_rich():
    samples=[sample(0,4,0),sample(1,12,1),sample(2,9,8),sample(3,11,0),sample(4,2,0)]
    assert [int(s['packet']['source_frame']) for s in select_samples(samples)]==[0,1,2,3]
    samples[0]['role']='diagnostic'
    with pytest.raises(ValueError):select_samples(samples)


def test_normalization_is_original_formula_and_shape_guard():
    raw=np.ones((2,64,64,64),np.float32)*3
    np.testing.assert_array_equal(normalize_pair(raw,1.,5.),np.maximum((raw-1.)/(4.+1e-6),0))
    with pytest.raises(ValueError):normalize_pair(raw[:1],1.,5.)


def test_image_access_rejects_undeclared_movie_before_import_or_io(tmp_path):
    images=ImagePairs(tmp_path,['declared'])
    with pytest.raises(ValueError):images.get(sample(0,2,0))
