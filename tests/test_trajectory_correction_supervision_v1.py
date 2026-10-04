import numpy as np
from research.trajectory_correction_supervision_v1 import labels


def graph(p):return dict(edges=[] if p is None else [dict(source_id=p,target_id=20)])


def test_known_improvement_and_regression():
    groups=dict(children=np.array([20]),parents=np.array([10,11]),offsets=np.array([0,2]))
    parts=[dict(removed=((10,20),),added=((11,20),))]
    positive=labels(parts,graph(10),graph(11),groups,np.array([11]),np.array([True,True]))[0]
    negative=labels(parts,graph(10),graph(11),groups,np.array([10]),np.array([True,True]))[0]
    assert positive['label']==1 and positive['delta_correct_links']==1
    assert negative['label']==0 and negative['delta_correct_links']==-1


def test_unannotated_child_is_unknown_not_birth():
    groups=dict(children=np.array([20]),parents=np.array([10]),offsets=np.array([0,1]))
    row=labels([dict(removed=((10,20),),added=())],graph(10),graph(None),groups,
               np.array([-1]),np.array([False]))[0]
    assert row['label']==-1 and row['unknown_children']==1


def test_nearby_unmatched_alternative_is_ambiguous():
    groups=dict(children=np.array([20]),parents=np.array([10,11]),offsets=np.array([0,2]))
    row=labels([dict(removed=((10,20),),added=((11,20),))],graph(10),graph(11),groups,
               np.array([10]),np.array([True,False]))[0]
    assert row['label']==-1 and row['ambiguous_children']==1


def test_missing_known_parent_is_real_edge_error():
    groups=dict(children=np.array([20]),parents=np.array([10]),offsets=np.array([0,1]))
    row=labels([dict(removed=(),added=((10,20),))],graph(None),graph(10),groups,
               np.array([10]),np.array([True]))[0]
    assert row['label']==1
