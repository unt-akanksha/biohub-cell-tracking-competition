"""Torch tests to execute in the next small GPU probe, not yet GPU-verified."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from edge_feature_tta import D4_VIEWS,transform
from backward_flow_spatial_tta import invert_vector_field,install_backward_flow_spatial_tta


def transformed_vectors(field,rotation,reflected):
    value=transform(field,rotation,reflected)
    z,y,x=value.unbind(1)
    for _ in range(rotation): y,x=-x,y
    if reflected: x=-x
    return torch.stack([z,y,x],1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_rectangular_spatially_varying_vector_roundtrip(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    field=torch.arange(252.,device=device).reshape(1,3,3,4,7)
    for rotation,reflected in D4_VIEWS:
        moved=transformed_vectors(field,rotation,reflected)
        torch.testing.assert_close(invert_vector_field(moved,rotation,reflected),field,rtol=0,atol=0)


class BiasedFlow(torch.nn.Module):
    def __init__(self): super().__init__(); self.calls=0
    def forward(self,images):
        self.calls+=1
        values=images.new_tensor([2.,3.,4.]).reshape(1,3,1,1,1)
        return values.expand(images.shape[0],3,*images.shape[2:])


def test_fixed_direction_bias_cancels_but_z_is_unchanged():
    model=BiasedFlow().eval(); receipt=install_backward_flow_spatial_tta(model)
    result=model(torch.ones(1,2,3,4,7))
    expected=torch.tensor([2.,0.,0.]).reshape(1,3,1,1,1).expand_as(result)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    assert model.calls==8 and receipt['forward_calls']==1
    assert receipt['maximum_mean_absolute_flow_delta_um']>0
    with pytest.raises(ValueError): install_backward_flow_spatial_tta(model)
    model.train()
    with pytest.raises(ValueError): model(torch.ones(1,2,3,4,7))
