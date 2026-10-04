from copy import deepcopy
import pytest
from research.trajectory_event_release_v1 import check_event_stage


def fixture():
    nodes={str(i):dict(node_id=i,t=t,z=1,y=1,x=i+1) for i,t in enumerate([0,1,1,0,1])}
    initial=dict(nodes=nodes,edges=[])
    base=dict(nodes=nodes,edges=[dict(source_id=0,target_id=1),dict(source_id=3,target_id=4)])
    final=dict(nodes=deepcopy(nodes),edges=[dict(source_id=3,target_id=4)])
    details=dict(added_edges=0,removed_edges=1,nodes_and_coordinates_unchanged=True,
        processed_frames=1,solver_fallbacks=0,
        frames=[dict(t=1,fallback=False,added_edges=0,removed_edges=1)])
    return initial,base,final,details


def test_valid_event_death():
    assert check_event_stage(*fixture(),frames=2)['removed_edges']==1


def test_moved_cell_rejected():
    args=fixture();args[2]['nodes']['1']['x']=7
    with pytest.raises(ValueError,match='changed cells'):check_event_stage(*args,frames=2)


def test_existing_division_protected():
    args=fixture();args[1]['edges'].append(dict(source_id=0,target_id=2))
    with pytest.raises(ValueError,match='Protected'):check_event_stage(*args,frames=2)


def test_false_edit_count_rejected():
    args=fixture();args[3]['removed_edges']=0
    with pytest.raises(ValueError,match='edit counts'):check_event_stage(*args,frames=2)


def test_fallback_and_unprocessed_changes_rejected():
    args=fixture();args[3]['frames'][0]['fallback']=True;args[3]['solver_fallbacks']=1
    with pytest.raises(ValueError,match='Fallback transition'):check_event_stage(*args,frames=2)
    args=fixture();args[3]['frames']=[];args[3]['processed_frames']=0
    with pytest.raises(ValueError,match='Unprocessed'):check_event_stage(*args,frames=2)
