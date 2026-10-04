from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from owned_detector_logit_targets import peaks,targets,contract
from owned_detector_pu import targets as probability_targets


def test_strict_raw_peak_survives_even_when_all_probabilities_saturate():
    logits=np.full((5,5,5),20.,np.float32); logits[2,2,2]=30.
    assert np.all(expit(logits)==1)
    found=peaks(logits)
    # The true30 peak is retained; no20 plateau neighbor within radius1 can
    # become a maximum merely because probabilities rounded to the same1.
    assert any(np.array_equal(p,[2,2,2]) for p in found.coords)
    assert not any(np.array_equal(p,[2,2,1]) for p in found.coords)


def test_monotone_large_logit_ramp_has_only_its_true_corner_maximum():
    z,y,x=np.mgrid[:5,:5,:5]
    logits=(20+z+y+x).astype(np.float32)
    assert np.all(expit(logits)==1)
    np.testing.assert_array_equal(peaks(logits).coords,[[4,4,4]])


def test_unsaturated_targets_match_the_tested_probability_builder():
    native=np.full((13,13,13),-8.,np.float32); aligned=native.copy()
    native[6,6,6]=5.; aligned[6,6,7]=4.; native[3,3,3]=0.
    annotation=[[9.,9.,9.]]
    actual=targets(native,aligned,annotation)
    expected=probability_targets(expit(native),expit(aligned),annotation)
    for key in ('heatmap','weights','positive_mask','background_mask','unknown_mask','positive_coords'):
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-5,atol=1e-6)


def test_unknown_support_and_exact_annotation_override():
    native=np.full((13,13,13),-8.,np.float32); aligned=native.copy(); native[3,3,3]=0.
    result=targets(native,aligned,[[9,9,9]])
    assert result['unknown_mask'][3,3,3] and result['weights'][3,3,3]==0
    assert result['heatmap'][9,9,9]==1 and result['background_mask'][0,0,0]
    assert not contract()['graph_count_or_selection_score_used']


@pytest.mark.parametrize('point',[[-1,2,2],[3,3,13]])
def test_outside_annotations_rejected(point):
    logits=np.zeros((13,13,13),np.float32)
    with pytest.raises(ValueError): targets(logits,logits,[point])
