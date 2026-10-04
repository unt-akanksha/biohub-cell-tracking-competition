import numpy as np
from research.native_division_data_v3 import triplets,legacy_bindings


def test_true_siblings_and_known_other_parent_negatives():
    positions={0:np.array([0.,0,0]),1:np.array([1.,0,0]),2:np.array([-1.,0,0]),
               3:np.array([12.,0,0]),4:np.array([11.,0,0])}
    b=[dict(parent=10,child=11,parent_patch=0,child_patch=1,true_parent_um=positions[0]),
       dict(parent=10,child=12,parent_patch=0,child_patch=2,true_parent_um=positions[0]),
       dict(parent=20,child=21,parent_patch=3,child_patch=4,true_parent_um=positions[3])]
    p,used,c=triplets(b,positions)
    assert c['positive']==1 and c['negative']==4
    assert len(set(map(tuple,p['triples'])))==len(p['triples'])
    assert np.all(p['daughter_parents'][p['labels']==1,0]==p['daughter_parents'][p['labels']==1,1])
    assert np.all(p['daughter_parents'][p['labels']==0,0]!=p['daughter_parents'][p['labels']==0,1])


def test_near_other_parent_is_not_a_negative():
    pos={0:np.array([0.,0,0]),1:np.array([1.,0,0]),2:np.array([5.,0,0]),3:np.array([6.,0,0])}
    b=[dict(parent=10,child=11,parent_patch=0,child_patch=1,true_parent_um=pos[0]),
       dict(parent=20,child=21,parent_patch=2,child_patch=3,true_parent_um=pos[2])]
    p,used,c=triplets(b,pos)
    assert not len(p['labels']) and c['ambiguous_pairs_omitted']==2


def test_legacy_gt_identity_must_be_unique():
    movie=dict(nodes=[[10,0,0,0,0],[11,1,1,0,0],[20,0,12,0,0],[21,1,1,0,0]],edges=[[10,11],[20,21]])
    packet=dict(coords=np.array([[[1.625,0,0],[0.,0,0]]]),targets=np.array([0]),ids=np.array([[1,0]]))
    b,pos,c=legacy_bindings(packet,movie,0)
    assert not b and c['identity_omitted']==1
