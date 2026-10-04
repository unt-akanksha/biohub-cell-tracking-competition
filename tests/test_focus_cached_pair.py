from copy import deepcopy
import numpy as np
import pytest
from research.focus_cached_pair import validate_pair


def example():
    coords=np.array([[0,1.2,2.3,3.4],[0,4.5,5.6,6.7],[1,1.3,2.4,3.5],[1,4.6,5.7,6.8]],np.float32)
    p=dict(source_frame=np.array(0,dtype=np.int64),source_indices=np.array([0,1],np.int64),target_indices=np.array([2,3],np.int64),
        source_coords=coords[:2,1:],target_coords=coords[2:,1:],source_features=np.zeros((2,32),np.float32),target_features=np.zeros((2,32),np.float32),
        source_pos=np.zeros((2,32),np.float32),target_pos=np.zeros((2,32),np.float32),backward_um=np.zeros((2,3),np.float32),labels=np.array([0,2],np.int64))
    return coords,p


def test_round_trip_and_supervision_counts(tmp_path):
    coords,p=example();expected=validate_pair(p,coords)
    assert expected==dict(source_frame=0,source_nodes=2,target_nodes=2,known_parent=1,known_absent=1,unknown=0)
    path=tmp_path/'pair.npz';np.savez_compressed(path,**p)
    with np.load(path,allow_pickle=False) as data:assert validate_pair(dict(data),coords)==expected


@pytest.mark.parametrize('field',['source_indices','target_indices','source_coords','target_coords'])
def test_reordered_nodes_cannot_silently_change_labels(field):
    coords,p=example();p=deepcopy(p);p[field]=p[field][::-1]
    with pytest.raises(ValueError):validate_pair(p,coords)


def test_invalid_labels_features_and_frame():
    coords,p=example()
    for key,value in [('labels',np.array([0,3],np.int64)),('source_features',np.full((2,32),np.nan,np.float32)),('source_frame',np.array(.5))]:
        changed=deepcopy(p);changed[key]=value
        with pytest.raises(ValueError):validate_pair(changed,coords)


def test_unknown_labels_do_not_get_counted_as_negative():
    coords,p=example();p['labels']=np.array([-1,-1],np.int64)
    result=validate_pair(p,coords)
    assert result['unknown']==2 and result['known_parent']==result['known_absent']==0
