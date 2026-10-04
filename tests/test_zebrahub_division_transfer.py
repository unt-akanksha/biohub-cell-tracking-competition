import numpy as np
from research.zebrahub_division_transfer import aligned_biohub,aligned_external,aligned_context,geometry_and_anchors,external_examples,shared_geometry
from research.image_division_context import image_context


def test_actual_frame_alignment_not_target_centered_time():
    p=np.zeros((1,3,3,17,17,17),dtype=np.float16)
    p[:,:,0]=9;p[:,:,1]=10;p[:,:,2]=11
    source=p[:,0].copy();target=np.zeros((2,3,17,17,17),np.float16)
    target[:,0]=10;target[:,1]=11;target[:,2]=12
    np.testing.assert_array_equal(aligned_biohub(p),aligned_external(source,target,[[0,0,1]]))
    assert aligned_external(source,target,[[0,0,1]])[0,:,0,0,0,0].tolist()==[10,10,10]


def test_aligned_context_exactly_matches_image_recomputation():
    rng=np.random.default_rng(4);patch=rng.normal(size=(3,17,17,17)).astype(np.float16)
    _,anchors,_=geometry_and_anchors([0,0,0],[0,-4,0],[0,4,0])
    c,m=image_context(patch,anchors);actual_c,actual_m=aligned_context(c[None],m[None])
    expected_c,expected_m=image_context(patch[[1,1,2]],anchors)
    np.testing.assert_array_equal(actual_c[0],expected_c.astype(np.float16))
    np.testing.assert_array_equal(actual_m[0],expected_m)


def test_true_pair_once_and_known_children_never_negatives():
    data=dict(source_ids=np.array([10]),target_ids=np.array([20,30,40,50]),source_coords_um=np.zeros((1,3)),
        target_coords_um=np.array([[0,-4,0],[0,4,0],[0,5,1],[1,3,0]]),
        positive_mask=np.array([[True,True,False,False]]),division_target=np.array([1.]))
    rows=external_examples(data)
    assert sum(r['target'] for r in rows)==1
    assert rows[0]['indices']==(0,0,1)
    assert all(r['indices'][2] in (2,3) for r in rows if not r['target'])
    assert len(rows)==3


def test_geometry_has_no_external_domain_identity_from_velocity():
    g,_,_=geometry_and_anchors([0,0,0],[0,-4,0],[0,4,0]);g[7:]=[1,2]
    actual=shared_geometry(g[None]);assert np.isnan(actual[0,7:]).all()
    np.testing.assert_array_equal(actual[0,:7],g[:7])


def test_external_replication_does_not_increase_its_domain_weight():
    from research.zebrahub_transfer_head import fit_transfer
    from research.frozen_image_head import predict
    rng=np.random.default_rng(40)
    def packet(offset):
        x=rng.normal(size=(10,1346))+offset;y=(np.arange(10)%2).astype(float)
        x[:,0]+=y*2
        return dict(features=x,targets=y,eligible=np.ones(10,dtype=bool),weights=np.ones(10))
    external=packet(2);biohub=packet(-2)
    first=fit_transfer(external,biohub)
    doubled={k:np.concatenate([v,v]) for k,v in external.items()}
    second=fit_transfer(doubled,biohub)
    np.testing.assert_allclose(first['mean'],second['mean'],atol=1e-12)
    np.testing.assert_allclose(first['scale'],second['scale'],atol=1e-12)
    np.testing.assert_allclose(predict(biohub['features'],first),predict(biohub['features'],second),atol=1e-7)
    assert first['external_domain_mass']==first['biohub_domain_mass']==.5
