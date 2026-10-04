from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from owned_detector_pu import targets,contract


def test_forced_annotation_and_unknown_teacher_support_remain_distinct():
    native=np.zeros((13,13,13),np.float32); aligned=native.copy()
    native[3,3,3]=.5
    result=targets(native,aligned,[[9,9,9]])
    assert result['forced_annotation_count']==1 and result['consensus_count']==0
    assert result['positive_mask'][9,9,9] and result['heatmap'][9,9,9]==1
    assert result['unknown_mask'][3,3,3] and result['weights'][3,3,3]==0
    assert result['background_mask'][0,0,0] and result['weights'][0,0,0]==pytest.approx(.01)


def test_gaussian_tails_do_not_turn_background_into_positive_loss_samples():
    native=np.zeros((11,11,11),np.float32)
    result=targets(native,native,[[5,5,5]])
    assert result['background_mask'][8,5,5]
    assert result['heatmap'][8,5,5]==0
    assert np.array_equal(result['heatmap']>0,result['positive_mask'])


def test_view_consensus_is_one_to_one_and_annotations_take_precedence():
    native=np.zeros((13,13,13),np.float32); aligned=native.copy()
    native[6,6,6]=.99; aligned[6,6,7]=.98
    result=targets(native,aligned,[[6,6,6.25]])
    assert result['consensus_count']==1 and result['forced_annotation_count']==1
    np.testing.assert_array_equal(result['positive_coords'],[[6,6,6.25]])


@pytest.mark.parametrize('point',[[-.1,3,3],[3,3,13],[float('nan'),3,3]])
def test_invalid_annotation_never_silently_clipped(point):
    native=np.zeros((13,13,13),np.float32)
    with pytest.raises(ValueError): targets(native,native,[point])


def test_frozen_policy_uses_no_graph_counts_or_selection_scores():
    assert contract()['graph_count_or_selection_score_used'] is False
    assert contract()['background_weight']==.01
