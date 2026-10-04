import numpy as np
from research.native_joint_linking_v2 import joint_repair


def packet(mapping):
    ids=np.full((len(mapping),17),-1,np.int64);p=np.zeros((len(mapping),17))
    for row,(child,parent) in enumerate(mapping.items()):ids[row,:2]=[child,parent];p[row,0]=.995;p[row,16]=.005
    return ids,p


def test_joint_cycle_preserves_node_edge_counts():
    graph=dict(nodes={str(i):dict(node_id=i,t=0 if i<3 else 1,z=1,y=i,x=1) for i in (1,2,3,4)},
               edges=[dict(source_id=1,target_id=3),dict(source_id=2,target_id=4)])
    result,events=joint_repair(graph,*packet({3:2,4:1}))
    assert result['nodes']==graph['nodes'] and len(result['edges'])==2
    assert {(e['source_id'],e['target_id']) for e in result['edges']}=={(2,3),(1,4)}
    assert len(events)==1 and events[0]['kind']=='joint_identity_cycle'
    unchanged,events=joint_repair(graph,*packet({3:2,4:2}))
    assert not events and unchanged==graph


def test_supported_division_requires_parent_and_both_daughter_persistence():
    times={0:0,1:1,2:2,3:3,4:4,5:5,6:3,7:4,8:5}
    graph=dict(nodes={str(i):dict(node_id=i,t=t,z=1,y=i,x=1) for i,t in times.items()},
        edges=[dict(source_id=a,target_id=b) for a,b in [(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8)]])
    result,events=joint_repair(graph,*packet({3:2,6:2}))
    assert len(events)==1 and events[0]['kind']=='persistent_division'
    assert dict(source_id=2,target_id=6) in result['edges'] and result['nodes']==graph['nodes']
    graph['edges'].remove(dict(source_id=7,target_id=8))
    unchanged,events=joint_repair(graph,*packet({3:2,6:2}))
    assert not events and unchanged==graph
