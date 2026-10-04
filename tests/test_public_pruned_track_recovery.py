import copy
from research.public_pruned_track_recovery import recover

CONFIG=dict(SHORT_TRACK_RESCUE_MIN_LEN=4,OUTPUT_MIN_TRACK_LEN=6,
            SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB=.88,SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM=3.,
            SHORT_TRACK_RESCUE_MAX_NODES_ABS=120,SHORT_TRACK_RESCUE_MAX_NODES_FRAC=.012)


def example(prob=.95,distance=1.):
    nodes={str(i):dict(node_id=i,t=i,z=20,y=30,x=40) for i in range(4)}
    edges=[dict(source_id=i,target_id=i+1,edge_prob=prob,distance_um=distance) for i in range(3)]
    final=dict(nodes={'99':dict(node_id=99,t=0,z=40,y=40,x=40)},edges=[])
    return final,dict(nodes=nodes,edges=[]),edges


def run(data,count=1000):
    return recover(*data,CONFIG,lambda nodes,edges,stats:nodes,count)


def test_only_strong_real_nodes_recovered_without_touching_parent():
    data=example();old=copy.deepcopy(data)
    graph,receipt=run(data)
    assert data==old
    assert receipt['added_nodes']==4 and receipt['added_edges']==3
    assert graph['nodes']['99']==old[0]['nodes']['99']
    assert set(graph['nodes'])=={'0','1','2','3','99'}
    assert not receipt['invented_detections'] and not receipt['authorized_for_submission']


def test_weak_fast_or_overbudget_components_not_recovered():
    for data,count in ((example(prob=.87),1000),(example(distance=3.1),1000),(example(),100)):
        graph,receipt=run(data,count)
        assert graph==data[0] and receipt['added_nodes']==0


def test_component_touching_retained_graph_is_not_recovered():
    data=example();data[0]['nodes']['0']=copy.deepcopy(data[1]['nodes']['0'])
    graph,receipt=run(data)
    assert graph==data[0] and receipt['added_nodes']==0
