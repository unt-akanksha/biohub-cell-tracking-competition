import numpy as np
import pytest
from research.native_graph_repair_v2 import frame_candidates,propose_repairs,apply_repairs,select_model_groups


def graph():
    positions=[(1,0,10),(2,0,12),(3,1,11),(4,1,13)]
    return dict(nodes={str(i):dict(node_id=i,t=t,z=20,y=y,x=20) for i,t,y in positions},edges=[dict(source_id=1,target_id=3)])


def probabilities(packet,child,parent):
    result=np.zeros((len(packet['ids']),17));result[:,-1]=1
    row=np.flatnonzero(packet['ids'][:,0]==child)[0]
    col=np.flatnonzero(packet['ids'][row,1:]==parent)[0]
    result[row,-1]=.005;result[row,col]=.995
    return result


def test_addition_has_no_new_nodes_or_fork():
    original=graph();packet=frame_candidates(original,0)
    proposed=propose_repairs(original,packet,probabilities(packet,4,2))
    result,accepted=apply_repairs(original,proposed)
    assert len(accepted)==1 and result['nodes']==original['nodes']
    assert len(result['edges'])==2 and original['edges']==[dict(source_id=1,target_id=3)]


def test_confident_rewire_and_existing_division_protection():
    original=graph();packet=frame_candidates(original,0)
    posterior=probabilities(packet,3,2)
    result,accepted=apply_repairs(original,propose_repairs(original,packet,posterior))
    assert accepted[0]['kind']=='rewire' and result['edges']==[dict(source_id=2,target_id=3)]
    original['edges'].append(dict(source_id=1,target_id=4))
    assert not propose_repairs(original,packet,posterior)


def test_competing_children_get_one_parent_only():
    original=graph();packet=frame_candidates(original,0)
    a=probabilities(packet,3,2);b=probabilities(packet,4,2)
    a[1]=b[1]
    result,accepted=apply_repairs(original,propose_repairs(original,packet,a))
    assert len(accepted)==1
    assert sum(e['source_id']==2 for e in result['edges'])==1


def test_reject_unnormalized_and_padded_probability():
    original=graph();packet=frame_candidates(original,0)
    with pytest.raises(ValueError):propose_repairs(original,packet,np.ones((2,17)))
    p=np.zeros((2,17));p[:,15]=1
    with pytest.raises(ValueError):propose_repairs(original,packet,p)


def test_component_admission_has_no_transformer_only_fallback():
    training=dict(members=[dict(embryo=e,family=f,source_gate=True,weights_sha256=e+f)
                    for e in ('6bba','44b6') for f in ('resnet3d','token_transformer3d')])
    cross=dict(folds=[dict(source=e,status='evaluated_once',methods={f:dict(conditional_gate=True)
                    for f in ('resnet3d','token_transformer3d','fixed_equal_mixture')}) for e in ('6bba','44b6')])
    selected=select_model_groups(training,cross)
    assert len(selected)==4 and all(m['weight']==.25 for m in selected)
    cross['folds'][1]['methods']['resnet3d']['conditional_gate']=False
    selected=select_model_groups(training,cross)
    assert len(selected)==2 and all(m['weight']==.5 and m['embryo']=='6bba' for m in selected)
    cross['folds'][0]['methods']['token_transformer3d']['conditional_gate']=False
    selected=select_model_groups(training,cross)
    assert len(selected)==1 and selected[0]['family']=='resnet3d' and selected[0]['weight']==1.
