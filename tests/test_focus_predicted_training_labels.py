import numpy as np
import pytest
from research.focus_predicted_training_labels import targets,role


def test_known_parent_unknown_target_and_birth():
    coords=np.array([[0,1,2,3],[1,1,2,3],[1,2,2,3],[1,3,2,3]],float)
    truth={10:[0,1,2,3],11:[1,1,2,3],12:[1,3,2,3]}
    row=targets(coords,{0:10,1:11,2:None,3:12},truth,[(10,11)],0)
    np.testing.assert_array_equal(row['labels'],[0,-1,-1])
    assert row['status']==['known_parent','unknown_target','no_unique_adjacent_annotated_parent']


def test_null_requires_geometric_absence_not_failed_matching():
    coords=np.array([[0,1,2,3],[1,1,2,3]],float)
    truth={10:[0,1,2,3],11:[1,1,2,3]}
    row=targets(coords,{0:None,1:11},truth,[(10,11)],0)
    assert row['labels'][0]==-1
    truth[10]=[0,20,2,3]
    row=targets(coords,{0:None,1:11},truth,[(10,11)],0)
    assert row['labels'][0]==row['null_index']==1


def test_exact_radius_boundary_is_unknown():
    truth={10:[0,0,0,0],11:[1,0,0,0]}
    points=np.array([[0,7/1.625,0,0],[1,0,0,0]])
    assert targets(points,{0:None,1:11},truth,[(10,11)],0)['labels'][0]==-1


def test_division_daughters_both_supervised_without_fake_divisions():
    coords=np.array([[0,1,1,1],[1,1,1,1],[1,2,1,1]])
    truth={10:coords[0],11:coords[1],12:coords[2]}
    np.testing.assert_array_equal(targets(coords,{0:10,1:11,2:12},truth,[(10,11),(10,12)],0)['labels'],[0,0])


def test_ambiguous_parent_and_target_ignored():
    coords=np.array([[0,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1]])
    truth={10:coords[0],11:coords[2]}
    row=targets(coords,{0:10,1:10,2:11,3:11},truth,[(10,11)],0)
    assert all(v==-1 for v in row['labels'])


def test_frame_partitions_have_no_shared_frames():
    fit={f for t in range(99) if role(t)=='fitting' for f in (t,t+1)}
    diagnostic={f for t in range(99) if role(t)=='diagnostic' for f in (t,t+1)}
    assert fit==set(range(70)) and diagnostic==set(range(80,100)) and not fit&diagnostic
    with pytest.raises(ValueError):role(99)


def test_malformed_matches_and_empty_parent_set():
    truth={10:[0,1,1,1],11:[1,1,1,1]};coords=np.array([[1,1,1,1]])
    assert targets(coords,{0:11},truth,[(10,11)],0)['labels'][0]==0
    with pytest.raises(ValueError):targets(coords,{},truth,[(10,11)],0)
    with pytest.raises(ValueError):targets(coords,{0:10},truth,[(10,11)],0)
