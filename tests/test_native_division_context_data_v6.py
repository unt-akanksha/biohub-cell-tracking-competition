import numpy as np
import pytest
from research.native_division_context_data_v6 import context_layout


def packet():
    return dict(patches=np.zeros((3,3,15,15,15),np.float16),triples=np.array([[0,1,2]]),
                labels=np.array([1]),coords=np.array([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]]))


def test_distinct_pre_and_post_frames():
    result=context_layout(packet(),10)
    np.testing.assert_array_equal(result['context_times'],[9,12,12])
    np.testing.assert_array_equal(result['context_valid'],[True,True,True])


def test_boundary_is_masked_not_dropped_or_fabricated():
    np.testing.assert_array_equal(context_layout(packet(),0)['context_times'],[-1,2,2])
    np.testing.assert_array_equal(context_layout(packet(),98)['context_times'],[97,-1,-1])


def test_ambiguous_patch_role_rejected():
    p=packet();p['triples']=np.array([[0,0,2]])
    with pytest.raises(ValueError,match='identity'):context_layout(p,10)


def test_duplicate_patch_coordinates_must_agree():
    p=packet();p['triples']=np.repeat(p['triples'],2,axis=0);p['coords']=np.repeat(p['coords'],2,axis=0);p['labels']=np.array([1,1])
    p['coords'][1,0,0]+=1
    with pytest.raises(ValueError,match='identity'):context_layout(p,10)
