import numpy as np
from research.focus_feature_separation import compare,summarize


def test_known_parent_only_and_exact_matching_features():
    p=dict(labels=np.array([0,-1,2]),source_coords=np.array([[0.,0,0],[1.,0,0]]),target_coords=np.zeros((3,3)),
           backward_um=np.zeros((3,3)),source_features=np.array([[1.,0],[0.,1]]),target_features=np.array([[1.,0],[0.,1],[1.,1]]))
    values=compare(p,dict(mean_um=[0,0,0],variance_um2=[1,1,1]));result=summarize(values)
    assert result['pairs']==1 and result['raw_cosine_pairwise_accuracy']==1 and result['physical_pairwise_accuracy']==1
    assert result['mean_true_cosine']==1 and result['mean_wrong_cosine']==0


def test_identical_features_are_ties_not_wins():
    values=np.ones((4,8));result=summarize(values)
    assert result['raw_cosine_pairwise_accuracy']==.5 and result['l2_pairwise_accuracy']==.5
