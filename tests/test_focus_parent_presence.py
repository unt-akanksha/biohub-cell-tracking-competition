import numpy as np
import pytest
from scipy.special import logsumexp
from research.focus_parent_presence import summarize,metrics,fit,FEATURES


def test_null_factorization_exact_and_unknown_ignored():
    scores=np.array([[1.,-8.,-2.,4.],[0.,-9.,0.,2.]])
    packet=dict(source_features=np.ones((2,32)),target_features=np.ones((4,32)),source_coords=np.array([[0,0,0],[1,2,3]]),labels=np.array([0,2,1,-1]),target_indices=np.arange(4))
    data=summarize(packet,scores,scores-1);m=metrics(data)
    augmented=np.vstack([scores,np.full(4,-4.5)]);labels=packet['labels'][:3]
    expected=logsumexp(augmented[:,:3],axis=0)-augmented[labels,np.arange(3)]
    assert m['loss_sum']==pytest.approx(expected.sum(),abs=1e-12)
    assert m['correct_parent']==2 and m['correct_absent']==1
    assert data['target_indices'].tolist()==[0,1,2]


def test_offset_fit_and_diagnostic_exclusion():
    rng=np.random.default_rng(17);x=rng.normal(size=(300,len(FEATURES)));y=(x[:,0]>.4).astype(int)
    data=dict(context=x,offset=np.zeros(300),present=y,conditional_nll=np.zeros(300),conditional_max=np.zeros(300),correct_if_present=np.ones(300))
    with pytest.raises(ValueError):fit(data,'diagnostic')
    model=fit(data,'fitting');assert metrics(data,model)['nll']<metrics(data)['nll']


def test_single_source_finite():
    p=dict(source_features=np.ones((1,32)),target_features=np.ones((1,32)),source_coords=np.zeros((1,3)),labels=np.array([0]),target_indices=np.array([4]))
    data=summarize(p,np.zeros((1,1)),np.zeros((1,1)))
    assert data['context'].shape==(1,7) and np.isfinite(data['context']).all()


def test_joint_null_wins_despite_majority_real_parent_mass():
    scores=np.array([[-4.7],[-4.8]])
    p=dict(source_features=np.ones((2,32)),target_features=np.ones((1,32)),source_coords=np.zeros((2,3)),labels=np.array([2]),target_indices=np.array([5]))
    data=summarize(p,scores,scores)
    assert data['offset'][0]>0  # Total real-parent mass exceeds the null mass.
    assert metrics(data)['correct_absent']==1  # But null is the largest class.
