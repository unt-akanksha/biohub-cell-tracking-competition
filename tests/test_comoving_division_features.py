import numpy as np
from research.comoving_division_features import features,WIDTH
from research.learned_division_recovery import VOXEL_SIZE_ZYX_UM


def scene(drift=np.zeros(3)):
    physical={1:(-2,np.array([-2.,0,0])),2:(-1,np.array([-1.,0,0])),3:(0,np.zeros(3)),
              4:(1,np.array([1.,-4,0])),5:(1,np.array([1.,4,0]))}
    nodes={k:(t,*((p+t*drift)/VOXEL_SIZE_ZYX_UM)) for k,(t,p) in physical.items()}
    return nodes,{2:[1],3:[2],4:[3],5:[3]}


def test_daughter_swap_and_no_other_graph_neighborhood_use():
    nodes,incoming=scene();a,meta=features(nodes,incoming,3,4,5)
    nodes[99]=(0,np.nan,np.nan,np.nan)
    b,_=features(nodes,incoming,3,5,4)
    np.testing.assert_array_equal(a,b)
    assert meta['available'] and a.shape==(WIDTH,)


def test_comoving_geometry_invariant_to_constant_tissue_drift():
    n,i=scene();_,a=features(n,i,3,4,5)
    n,i=scene(np.array([7.,-3.,2.]));_,b=features(n,i,3,4,5)
    np.testing.assert_allclose(a['comoving_geometry'],b['comoving_geometry'],atol=1e-12)
    assert not np.allclose(a['raw_geometry'],b['raw_geometry'])


def test_missing_or_ambiguous_history_is_explicitly_unavailable():
    nodes,incoming=scene();incoming[3]=[]
    x,m=features(nodes,incoming,3,4,5)
    np.testing.assert_array_equal(x,np.zeros(WIDTH));assert not m['available']
    nodes[9]=nodes[2];incoming[3]=[2,9]
    _,m=features(nodes,incoming,3,4,5);assert not m['available']


def test_features_do_not_use_the_labels_of_proposed_daughter_edges():
    nodes,incoming=scene();before,_=features(nodes,incoming,3,4,5)
    incoming.pop(4);incoming.pop(5)
    after,_=features(nodes,incoming,3,4,5)
    np.testing.assert_array_equal(before,after)
