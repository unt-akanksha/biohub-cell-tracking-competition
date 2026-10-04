import numpy as np
import pytest
from research.focus_motion_error_attribution import classify_unlinked


def test_recovered_and_missing_endpoint_not_called_association_failure():
    coords=np.array([[0,1,1,1],[1,1,1,1]],float)
    r=classify_unlinked(coords,np.zeros((2,3)),{0:10,1:11},[(0,1)],[(10,11),(11,12)])
    assert r['both_detected_unlinked']==0


def test_parent_competition_is_distinct_from_long_displacement():
    coords=np.array([[0,1,1,1],[0,1,1,1],[1,1,1,1]],float)
    r=classify_unlinked(coords,np.zeros((3,3)),{0:10,1:-1,2:11},[],[(10,11)])
    assert r['distance_beats_null_but_parent_competition']==1
    coords[2,2:]=100
    r=classify_unlinked(coords,np.zeros((3,3)),{0:10,2:11},[],[(10,11)])
    assert r['all_pairs_at_or_beyond_null_distance']==1


def test_above_threshold_missing_proposal_is_separated():
    coords=np.array([[0,1,1,1],[1,1,1,1]],float)
    r=classify_unlinked(coords,np.zeros((2,3)),{0:10,1:11},[],[(10,11)])
    assert r['posterior_above_half_but_topology_blocked']==1


def test_wrong_time_and_nonfinite_inputs_rejected():
    coords=np.array([[0,1,1,1],[2,1,1,1]],float)
    with pytest.raises(ValueError): classify_unlinked(coords,np.zeros((2,3)),{0:10,1:11},[],[(10,11)])
    coords[0,1]=np.nan
    with pytest.raises(ValueError): classify_unlinked(coords,np.zeros((2,3)),{},[],[])
