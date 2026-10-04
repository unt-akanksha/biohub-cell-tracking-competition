import copy
import numpy as np
import pytest
from research.focus_parent_dropout import augment, SCALE


def sample():
    coords = np.array([[0., 0., 0.], [0., 0., 20.], [0., 0., 100.]], np.float32)
    return dict(stem='fit', role='fitting', packet=dict(source_frame=np.array(0),
        source_coords=coords, source_indices=np.arange(3, dtype=np.int64),
        source_features=np.ones((3,32),np.float32), source_pos=np.ones((3,32),np.float32),
        target_coords=np.zeros((4,3),np.float32), labels=np.array([0,0,3,-1],np.int64)))


def test_relabels_division_null_and_unknown_without_mutation():
    original=sample(); before=copy.deepcopy(original); result=augment(original)
    np.testing.assert_array_equal(result['packet']['source_indices'], [2])
    np.testing.assert_array_equal(result['packet']['labels'], [1,1,1,-1])
    assert result['provenance']['synthetic_null_columns']==[0,1]
    for k,v in original['packet'].items(): np.testing.assert_array_equal(v,before['packet'][k])
    np.testing.assert_array_equal(result['packet']['target_coords'],before['packet']['target_coords'])
    assert augment(original)['provenance']==result['provenance']


def test_reject_diagnostic_before_packet_access():
    with pytest.raises(ValueError): augment(dict(role='diagnostic'))


def test_no_parent_and_all_sources_removed():
    s=sample();s['packet']['labels'][:]=3
    assert augment(s) is None
    s=sample();s['packet']['source_coords'][:]=0
    result=augment(s)
    assert not result['provenance']['eligible_nonempty_source']
    np.testing.assert_array_equal(result['packet']['labels'],[0,0,0,-1])


def test_retained_sources_outside_all_selected_gt_match_balls():
    s=sample(); result=augment(s);chosen=result['provenance']['selected_parent_row']
    predicted=s['packet']['source_coords'][chosen]*SCALE
    rng=np.random.default_rng(42); directions=rng.normal(size=(1000,3))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    possible_gt=predicted+7*directions
    remaining=result['packet']['source_coords']*SCALE
    assert np.linalg.norm(remaining[:,None]-possible_gt[None],axis=2).min()>7


def test_ambiguous_removed_parent_is_not_forced_null():
    s=sample();s['packet']['source_coords']=np.array([[0.,0.,0.],[0.,0.,30.],[0.,0.,60.]],np.float32)
    s['packet']['labels']=np.array([0,1,3,-1],np.int64)
    for i in range(100):
        s['stem']=str(i);result=augment(s)
        if result['provenance']['selected_parent_row']==0: break
    assert result['provenance']['selected_parent_row']==0
    assert result['packet']['labels'][0]==1
    assert result['packet']['labels'][1]==-1


def test_retained_parent_and_null_indices_are_distinct_after_drop():
    s=sample();s['packet']['labels']=np.array([0,2,3,-1],np.int64)
    for i in range(100):
        s['stem']=str(i);result=augment(s)
        if result['provenance']['selected_parent_row']==0:break
    assert result['provenance']['selected_parent_row']==0
    np.testing.assert_array_equal(result['packet']['labels'],[1,0,1,-1])
    np.testing.assert_array_equal(result['packet']['source_indices'],[2])
