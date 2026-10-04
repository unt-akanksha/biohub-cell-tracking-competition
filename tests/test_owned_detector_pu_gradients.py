from pathlib import Path
import sys
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from owned_detector_pu import targets,loss


def test_unknown_region_has_exactly_zero_gradient_and_positives_are_learned():
    native=np.zeros((13,13,13),np.float32); aligned=native.copy(); native[3,3,3]=.5
    result=targets(native,aligned,[[9,9,9]])
    logits=torch.zeros(13,13,13,requires_grad=True)
    objective=loss(logits,torch.from_numpy(result['heatmap']),torch.from_numpy(result['weights']))
    objective.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[torch.from_numpy(result['unknown_mask'])]==0).all()
    assert logits.grad[9,9,9]<0 and logits.grad[0,0,0]>0


def test_background_count_does_not_amplify_negative_weight():
    values=[]
    for n in (5,15):
        logits=torch.zeros(n,n,n,requires_grad=True); target=torch.zeros_like(logits)
        weights=torch.full_like(logits,.01); target[2,2,2]=1.; weights[2,2,2]=1.
        values.append(float(loss(logits,target,weights)))
    assert values[0]==pytest.approx(values[1],abs=1e-6)


def test_nonfinite_terms_rejected():
    logits=torch.zeros(3,3,3)
    with pytest.raises(ValueError): loss(logits*float('nan'),logits,torch.ones_like(logits))


def test_probability_conversion_preserves_high_confidence_peak_order():
    from owned_detector_pu import teacher_probabilities
    logits=torch.tensor([9.,10.,11.],dtype=torch.float16)
    assert logits.sigmoid().unique().numel()==1
    probabilities=teacher_probabilities(logits)
    assert probabilities.dtype==torch.float32 and probabilities.unique().numel()==3
    assert torch.all(probabilities[1:]>probabilities[:-1]) and (probabilities<1).all()
