from copy import deepcopy
from itertools import product
import numpy as np
import pytest
from research.trajectory_correction_gate_v1 import components,apply_mask,features,FEATURES


def graph(edges):
    nodes={str(i):dict(node_id=i,t=int(i>=3),z=0,y=0,x=i) for i in range(6)}
    return dict(nodes=nodes,edges=[dict(source_id=a,target_id=b) for a,b in edges])


def test_swap_is_atomic_and_independent_deletion_can_abstain():
    initial=graph([]);base=graph([(0,3),(1,4),(2,5)]);new=graph([(0,4),(1,3)])
    parts=components(initial,base,new)
    assert len(parts)==2 and sorted(len(p['removed']) for p in parts)==[1,2]
    for bits in product((False,True),repeat=2):
        out=apply_mask(initial,base,new,bits)
        assert out['nodes']==base['nodes']
        parents=[e['source_id'] for e in out['edges']]
        children=[e['target_id'] for e in out['edges']]
        assert len(children)==len(set(children)) and max(parents.count(p) for p in parents)<=2
    assert apply_mask(initial,base,new,[True,True])==new
    assert apply_mask(initial,base,new,[False,False])==base


def test_fork_addition_is_safe_when_unchanged_continuation_exists():
    initial=graph([]);base=graph([(0,3),(1,4)]);new=graph([(0,3),(0,4)])
    assert len(components(initial,base,new))==1
    assert apply_mask(initial,base,new,[True])==new


def test_existing_division_and_moved_cells_rejected():
    initial=graph([]);base=graph([(0,3),(0,4)]);new=graph([(0,3)])
    with pytest.raises(ValueError,match='protected'):components(initial,base,new)
    new=deepcopy(base);new['nodes']['0']['x']=100
    with pytest.raises(ValueError,match='cells'):components(initial,base,new)


def test_synthetic_and_bad_mask_rejected():
    initial=graph([]);del initial['nodes']['5']
    with pytest.raises(ValueError,match='protected'):components(initial,graph([(2,5)]),graph([]))
    with pytest.raises(ValueError,match='boolean'):apply_mask(graph([]),graph([(0,3)]),graph([]),[1])


def test_features_do_not_include_identity_or_time_and_require_real_evidence():
    parts=[dict(t=1,removed=((10,20),),added=((11,20),))]
    groups=dict(children=np.array([20]),parents=np.array([10,11]),offsets=np.array([0,2]))
    matrix=np.arange(36).reshape(2,18)
    x=features(parts,groups,matrix)
    assert x.shape==(1,len(FEATURES)) and np.isfinite(x).all()
    shifted=[dict(t=999,removed=((110,120),),added=((111,120),))]
    renamed=dict(children=np.array([120]),parents=np.array([110,111]),offsets=np.array([0,2]))
    np.testing.assert_array_equal(x,features(shifted,renamed,matrix))
    with pytest.raises(ValueError,match='lacks'):features(parts,renamed,matrix)


def test_empty_graph_change_has_empty_features_and_no_mutation():
    base=graph([(0,3)])
    assert components(base,base,base)==[]
    assert apply_mask(base,base,base,np.array([],bool))==base
    groups=dict(children=np.array([3]),parents=np.array([0]),offsets=np.array([0,1]))
    assert features([],groups,np.zeros((1,18))).shape==(0,len(FEATURES))


def test_random_birth_death_fork_mixtures_remain_valid():
    rng=np.random.default_rng(42)
    for _ in range(100):
        children=rng.permutation([3,4,5])
        base=graph([(p,int(c)) for p,c in zip(range(3),children) if rng.random()>.3])
        counts=[0,0,0];edges=[]
        for c in (3,4,5):
            p=int(rng.integers(-1,3))
            if p>=0 and counts[p]<2:edges.append((p,c));counts[p]+=1
        candidate=graph(edges);initial=graph([])
        parts=components(initial,base,candidate)
        for mask in product((False,True),repeat=len(parts)):
            result=apply_mask(initial,base,candidate,np.asarray(mask,dtype=bool))
            targets=[e['target_id'] for e in result['edges']]
            parents=[e['source_id'] for e in result['edges']]
            assert len(targets)==len(set(targets))
            assert all(parents.count(p)<=2 for p in parents)
