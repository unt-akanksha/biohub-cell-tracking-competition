import numpy as np
from research.native_correspondence_data_v2 import groups, match_queries, pack_groups, proposals


def test_match_is_unique_and_never_injects_truth():
    matches = match_queries(np.array([[0.,0,0],[.1,0,0],[20,0,0]]), np.array([[0.,0,0]]))
    assert matches == {0: 0}


def test_candidates_precede_labels_and_ambiguity_not_null():
    parents = np.array([[0.,0,0],[5.,0,0],[15.,0,0]])
    child = np.array([[0.,0,0]])
    records, counts = groups(parents, child, np.array([[0.,0,0]]), child)
    assert records[0][1].tolist() == [0,1,2]
    assert records[0][2].tolist() == [True,False,True]
    assert records[0][3] == 0 and counts['positive'] == 1
    omitted, counts = groups(parents, child, np.array([[9.,0,0]]), child)
    assert not omitted and counts['ambiguous_omitted'] == 1


def test_far_absent_parent_is_null_and_pack_retains_validity():
    parent = np.array([[0.,0,0]])
    child = np.array([[1.,0,0]])
    records, counts = groups(parent, child, np.array([[50.,0,0]]), child)
    packet, p, c = pack_groups(records, parent, child)
    assert counts['null'] == 1 and packet['targets'].tolist() == [16]
    assert packet['ids'][0,:2].tolist() == [1,0]
    assert packet['valid'].sum() == 2
    np.testing.assert_equal(p, parent); np.testing.assert_equal(c, child)


def test_pool_grid_centers_have_half_voxel_offset():
    image = np.zeros((64,64,64), np.float32); image[20,30,40] = 2
    actual = proposals(image)
    np.testing.assert_allclose(actual, [[32.5, 121.5*.40625, 161.5*.40625]])
