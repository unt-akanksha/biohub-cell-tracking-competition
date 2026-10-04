import numpy as np
import pytest
from research.trajectory_disagreement_data_v1 import FEATURES,extract,supervision


def node(t,x):return dict(t=t,z=0.,y=0.,x=x)


def test_sparse_supervision_has_neural_motion_neither_and_ambiguity():
    truth={10:node(0,0),11:node(1,1)}
    rec=[dict(child=2,neural_parent=0,motion_parent=1)]
    for a,b,label in ((0,40,0),(40,0,1),(40,70,2),(0,10,-1)):
        nodes={0:node(0,a),1:node(0,b),2:node(1,1)}
        labels,_,counts=supervision(rec,nodes,truth,[(10,11)])
        assert labels.tolist()==[label] and sum(counts.values())==1


def test_unannotated_child_is_not_negative():
    nodes={0:node(0,0),1:node(0,40),2:node(1,90)}
    labels,reasons,_=supervision([dict(child=2,neural_parent=0,motion_parent=1)],nodes,
                               {10:node(0,0),11:node(1,1)},[(10,11)])
    assert labels.tolist()==[-1] and reasons==['unannotated_or_unmatched_child']


def test_annotation_without_parent_edge_stays_unknown():
    nodes={0:node(0,0),1:node(0,40),2:node(1,1)}
    labels,reasons,_=supervision([dict(child=2,neural_parent=0,motion_parent=1)],nodes,
                               {10:node(0,0),11:node(1,1)},[])
    assert labels.tolist()==[-1] and reasons==['parent_annotation_unavailable']


def test_nonfinite_geometry_rejected_instead_of_silent_negative():
    nodes={0:node(0,float('nan')),1:node(0,40),2:node(1,1)}
    with pytest.raises(ValueError):
        supervision([dict(child=2,neural_parent=0,motion_parent=1)],nodes,
                    {10:node(0,0),11:node(1,1)},[(10,11)])


def test_feature_schema_without_truth_and_raw_identity_guard():
    nodes={str(i):dict(node_id=i,**node(t,x)) for i,(t,x) in enumerate([(0,0),(0,40),(1,1),(2,2)])}
    initial=dict(nodes=nodes,edges=[dict(source_id=0,target_id=2,edge_prob=.99),dict(source_id=2,target_id=3,edge_prob=.98)])
    final=dict(nodes=nodes,edges=[dict(source_id=1,target_id=2),dict(source_id=2,target_id=3)])
    coords=np.array([[n[k] for k in ('t','z','y','x')] for n in nodes.values()])
    edges=np.array([[0,2,.99,.4],[2,3,.98,.4]])
    records,features,counts=extract(initial,final,coords,edges)
    assert len(records)==1 and features.shape==(1,len(FEATURES))
    assert records[0]['motif']=='free_parent' and counts['parent_disagreement']==1
    assert features[0,FEATURES.index('motion_probability_known')]==0
    assert features[0,FEATURES.index('motion_probability')]==.5
    bad=coords.copy();bad[0,3]=3
    with pytest.raises(ValueError):extract(initial,final,bad,edges)
