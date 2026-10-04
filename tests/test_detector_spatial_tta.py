from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from detector_spatial_tta import install_detector_spatial_tta
from motion_residual import install_motion_residual
from image_motion_residual import install_image_motion_residual


class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls=0
    def encode(self,images):
        self.calls+=1
        # A directional boundary artifact whose D4 mean has a known value.
        ramp=torch.arange(images.shape[-1],device=images.device).to(images).view(1,1,1,1,-1)
        return images,[images[:,i:i+1]+ramp for i in range(2)]
    def predict_edges(self,*args):
        return torch.zeros(args[2].shape[0],args[2].shape[1],args[3].shape[1])


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls=0
    def forward(self,images):
        self.calls+=1; self.input=images.clone()
        return images.new_zeros(images.shape[0],3,*images.shape[2:])


def test_rectangular_d4_mean_and_native_features():
    model=Detector().eval(); receipt=install_detector_spatial_tta(model)
    images=torch.arange(120.).reshape(1,2,3,4,5)
    features,logits=model.encode(images)
    assert features is images and model.calls==8
    for i,value in enumerate(logits):
        torch.testing.assert_close(value,images[:,i:i+1]+1.75)
    assert receipt['maximum_mean_absolute_logit_delta']>0 and receipt['encode_calls']==1


def test_flow_sees_one_native_pair_not_last_tta_view():
    model=Detector().eval(); receipt=install_detector_spatial_tta(model)
    install_motion_residual(model)
    flow=Flow(); install_image_motion_residual(model,flow,inference=True,calibration=[0.,1.,-4.5])
    images=torch.arange(120.).reshape(1,2,3,4,5)
    model.encode(images)
    assert model.calls==8 and flow.calls==1
    torch.testing.assert_close(flow.input,images.half().float())
    coordinates=torch.tensor([[[1.,4.,4.]]]); mask=torch.ones(1,1,dtype=torch.bool)
    result=model.predict_edges(None,None,coordinates,coordinates,None,None,mask,mask).sigmoid()
    torch.testing.assert_close(result,torch.sigmoid(torch.tensor(4.5)).reshape(1,1,1))
    assert receipt['encode_calls']==1


def test_invalid_installation_order_and_training_fail_closed():
    model=Detector()
    with pytest.raises(ValueError): install_detector_spatial_tta(model)
    model.eval(); install_motion_residual(model); install_image_motion_residual(model,Flow())
    with pytest.raises(ValueError): install_detector_spatial_tta(model)
    model=Detector().eval(); install_detector_spatial_tta(model)
    with pytest.raises(ValueError): install_detector_spatial_tta(model)
    model.train()
    with pytest.raises(ValueError): model.encode(torch.ones(1,2,3,4,5))
