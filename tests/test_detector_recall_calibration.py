import numpy as np
import pytest
from research.detector_recall_calibration import fit_threshold


def test_preserves_parent_recall_with_missing_annotations_in_denominator():
    result=fit_threshold([True,True,False,False],[.8,.9,.7,0.],baseline_threshold=.5)
    assert result['required_matched']==2 and result['candidate_retained_matched']==2
    assert result['candidate_retained_recall']==.5 and result['parent_recall']==.5
    assert result['threshold']==float(np.nextafter(np.float32(.8),np.float32(-np.inf)))
    assert not result['authorized_for_submission'] and not result['independent_recall_guarantee']


def test_fp32_saturated_ties_are_retained_under_strict_comparator():
    result=fit_threshold([True,True],[1.,1.],baseline_threshold=.5)
    assert result['threshold']<1 and np.float32(result['threshold'])<np.float32(1)
    assert result['candidate_retained_matched']==2


def test_insufficient_recall_does_not_silently_reduce_the_requirement():
    result=fit_threshold([True,True],[.9,0.],baseline_threshold=.5)
    assert result['status']=='baseline_recall_requirement_unmet' and result['threshold'] is None


def test_small_loss_budget_is_discrete_and_never_rounded_up():
    assert fit_threshold([True]*199,[.9]*199,baseline_threshold=.5)['required_matched']==199
    assert fit_threshold([True]*200,[.9]*200,baseline_threshold=.5)['required_matched']==199


@pytest.mark.parametrize('parent,probs', [([],[]),([False],[.9]),([True],[float('nan')]),
                                        ([True],[1.1]),([1],[.9]),([True],[.4])])
def test_invalid_or_censored_calibration_inputs_rejected(parent,probs):
    with pytest.raises(ValueError): fit_threshold(parent,probs,baseline_threshold=.5)
