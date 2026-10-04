import numpy as np
import pytest

from research.trajectory_event_supervision_v1 import label, for_problem
from research.trajectory_event_features_v1 import features, FEATURES
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_candidate_inventory_v1 import candidates


def graph():
    return dict(nodes={'0':dict(t=0,z=20,y=20,x=20),'1':dict(t=0,z=20,y=20,x=60),
                       '2':dict(t=1,z=20,y=20,x=21),'3':dict(t=1,z=20,y=20,x=22)},
                edges=[dict(source_id=0,target_id=2)])


def test_sparse_labels_never_turn_unknown_daughter_into_negative():
    g=graph(); groups=candidates(g,g)
    truth={10:g['nodes']['0'],11:g['nodes']['2']}
    targets,safe,report=label(groups,g['nodes'],truth,[(10,11)],{0:10,2:11})
    assert targets.tolist()==[0,-1]
    a,b=groups['offsets'][1:3]
    assert not safe[a:b].any()
    assert report['unknown_child']==1


def test_missing_parent_remains_unknown_not_birth():
    g=graph(); groups=candidates(g,g)
    targets,_,report=label(groups,g['nodes'],{10:g['nodes']['0'],11:g['nodes']['2']},[(10,11)],{2:11})
    assert (targets==-1).all() and report['matched_parent_missing_or_outside_candidates']==1


def test_nonunique_matches_rejected():
    g=graph(); groups=candidates(g,g)
    with pytest.raises(ValueError,match='one-to-one'):
        label(groups,g['nodes'],{10:g['nodes']['0']},[],{0:10,1:10})


def test_prediction_only_event_features_and_partial_mapping():
    g=graph(); groups=candidates(g,g)
    case,=list(frames(g,g,groups))
    matrix=features(case,g,np.ones((len(groups['parents']),18)))
    assert matrix.shape==(len(case['options']),30)==(len(case['options']),len(FEATURES))
    fork=np.flatnonzero(case['options'][:,2]>=0)[0]
    assert (matrix[fork,:18]==2).all()
    assert matrix[fork,19]==1 and matrix[fork,25]==0 and matrix[fork,27]==0
    targets,safe,count=for_problem(case,np.array([0,-1]),np.zeros(len(groups['parents']),bool))
    assert targets.tolist()==[0,-1] and not safe.any()
    assert count['known_constraints_retained']==1
