import numpy as np
from research.comoving_division_head import fit,predict
from research.frozen_image_head import fit as old_fit,predict as old_predict


def test_control_exactly_reproduces_original_fixed_penalty_head():
    rng=np.random.default_rng(3)
    packet=dict(features=rng.normal(size=(16,1346)),motion_features=rng.normal(size=(16,14)),
        targets=(np.arange(16)%2).astype(float),eligible=np.ones(16,dtype=bool),weights=np.ones(16))
    a=fit(packet,False)
    b=old_fit(packet['features'],packet['targets'],packet['eligible'],packet['weights'],.01)
    np.testing.assert_array_equal(predict(packet,a),old_predict(packet['features'],b))


def test_motion_head_save_and_frozen_transform(tmp_path):
    rng=np.random.default_rng(12)
    packet=dict(features=rng.normal(size=(16,1346)),motion_features=rng.normal(size=(16,14)),
        targets=(np.arange(16)%2).astype(float),eligible=np.ones(16,dtype=bool),weights=np.ones(16))
    state=fit(packet,True);path=tmp_path/'head.npz';np.savez(path,**state)
    with np.load(path,allow_pickle=False) as a:loaded=dict(a)
    np.testing.assert_array_equal(predict(packet,state),predict(packet,loaded))
    assert len(state['coefficients'])+1==1361
