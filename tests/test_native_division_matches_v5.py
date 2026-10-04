import numpy as np
from research.native_division_matches_v5 import isolated_additive_matches


def test_isolated_extension_and_strict_preservation():
    truth=np.array([[0,0,0],[0,20,0]],np.float32)
    points=np.array([[0,1,0],[0,25,0]],np.float32)
    assert isolated_additive_matches(truth,points)=={0:0,1:1}


def test_ambiguous_neighbors_not_new_labels():
    truth=np.zeros((1,3),np.float32);points=np.array([[0,5,0],[0,6,0]],np.float32)
    assert isolated_additive_matches(truth,points)=={}


def test_competing_new_annotations_rejected():
    truth=np.array([[0,0,0],[0,10,0]],np.float32);points=np.array([[0,5,0]],np.float32)
    assert isolated_additive_matches(truth,points)=={}


def test_cannot_steal_a_strict_match():
    truth=np.array([[0,0,0],[0,6,0]],np.float32);points=np.array([[0,1,0]],np.float32)
    assert isolated_additive_matches(truth,points)=={0:0}
