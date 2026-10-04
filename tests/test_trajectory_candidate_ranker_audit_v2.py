import numpy as np
from research.trajectory_candidate_ranker_audit_v2 import predict,evaluate


def test_ambiguous_highest_scoring_alternative_is_not_hidden_from_inference():
    groups=dict(children=np.array([8]),parents=np.array([1,2,3]),offsets=np.array([0,3]),
                current=np.array([1]),neural=np.array([1]))
    # Candidate 2 could have been excluded from supervised loss as ambiguous.
    predictions=predict(np.array([[1.],[3.],[0.]]),groups,np.array([1.]))
    result=evaluate(predictions,groups,np.array([1]))
    assert predictions.tolist()==[2]
    assert result['learned_correct']==0 and result['learned_breaks']==1
    assert not result['annotation_mask_used_for_inference']
