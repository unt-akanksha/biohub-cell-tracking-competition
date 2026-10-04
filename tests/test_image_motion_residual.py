from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from image_motion_residual import install_image_motion_residual
from motion_residual import install_motion_residual,parent_probabilities


class Neural(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.2))

    def encode(self,images):
        return images,[images[:,0:1],images[:,1:2]]

    def predict_edges(self,*values):
        return self.bias.expand(values[2].shape[0],values[2].shape[1],values[3].shape[1])


class ConstantFlow(torch.nn.Module):
    def __init__(self,value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value,dtype=torch.float32))

    def forward(self,images):
        self.last_input = images.detach().clone()
        return self.value.reshape(1,3,1,1,1).expand(images.shape[0],3,*images.shape[2:])


def values():
    source = torch.tensor([[[1.,4.,0.],[1.,4.,8.]]])
    target = torch.tensor([[[1.,4.,4.]]])
    return (None,None,source,target,None,None,torch.ones(1,2,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool))


def test_zero_flow_preserves_detector_and_existing_scores_exactly():
    model = Neural()
    install_motion_residual(model)
    images = torch.ones(1,2,3,3,3)
    native_features,native_det = model.encode(images)
    native = model.predict_edges(*values())
    install_image_motion_residual(model,ConstantFlow([0,0,0]))
    features,det = model.encode(images)
    assert features is native_features
    for current,previous in zip(det,native_det):
        torch.testing.assert_close(current,previous,atol=0,rtol=0)
    torch.testing.assert_close(model.predict_edges(*values()),native,atol=0,rtol=0)


def test_backward_prior_gradients_and_frozen_flow():
    model = Neural(); flow = ConstantFlow([0,0,1.625])
    install_motion_residual(model)
    install_image_motion_residual(model,flow)
    model.encode(torch.ones(1,2,3,3,3))
    scores = model.predict_edges(*values())
    assert scores[0,1,0] > scores[0,0,0]
    scores.sum().backward()
    assert model.bias.grad.item() == 2
    assert flow.value.grad is None and not flow.value.requires_grad


def test_train_inference_agree_and_stale_pairs_fail_closed():
    train,infer = Neural(),Neural()
    for model,is_inference in [(train,False),(infer,True)]:
        install_motion_residual(model)
        install_image_motion_residual(model,ConstantFlow([0,0,1.625]),inference=is_inference)
        model.encode(torch.ones(1,2,3,3,3))
    expected = parent_probabilities(train.predict_edges(*values()))
    torch.testing.assert_close(infer.predict_edges(*values()).sigmoid(),expected)
    with pytest.raises(ValueError,match='fresh'):
        infer.predict_edges(*values())
    with pytest.raises(ValueError):
        install_image_motion_residual(Neural(),ConstantFlow([0,0,0]))


def test_real_out_of_grid_target_rejected():
    model = Neural()
    install_motion_residual(model)
    install_image_motion_residual(model,ConstantFlow([0,0,0]))
    model.encode(torch.ones(1,2,3,3,3))
    args = list(values())
    args[3] = torch.tensor([[[1.,4.,20.]]])
    with pytest.raises(ValueError,match='outside'):
        model.predict_edges(*args)


def test_embedded_flow_is_self_contained_frozen_and_hash_checked():
    from backward_flow_model import BackwardFlowNet
    from image_motion_residual import contract,flow_hash,embedded_flow
    flow = BackwardFlowNet()
    state = dict(identity=dict(image_motion=contract(),frozen_flow_sha256=flow_hash(flow)),
                 frozen_flow_model=flow.state_dict())
    restored = embedded_flow(state,'cpu')
    assert not any(p.requires_grad for p in restored.parameters())
    assert flow_hash(restored) == flow_hash(flow)
    state['frozen_flow_model']['head.bias'] = torch.ones(3)
    with pytest.raises(ValueError,match='checksum'):
        embedded_flow(state,'cpu')


def test_flow_input_quantization_matches_training_without_changing_detector():
    model = Neural(); flow = ConstantFlow([0,0,0])
    install_motion_residual(model)
    install_image_motion_residual(model,flow)
    images = torch.full((1,2,3,3,3),.1234567)
    features,_ = model.encode(images)
    assert features is images
    torch.testing.assert_close(flow.last_input,images.half().float(),rtol=0,atol=0)
    assert not torch.equal(flow.last_input,images)


@pytest.mark.parametrize('parameters',[[0.,1.,-4.5],[1.,1.,-4.5],[.3,.7,-2.]])
def test_calibrated_inference_matches_direct_parent_softmax(parameters):
    from independent_motion_prior import SCALE,VARIANCE
    model = Neural(); install_motion_residual(model)
    flow = ConstantFlow([0,0,1.625])
    install_image_motion_residual(model,flow,inference=True,calibration=parameters)
    images = torch.ones(1,2,3,3,3)
    features,det = model.encode(images)
    args = values()
    delta = (args[2].unsqueeze(-2)-args[3].unsqueeze(-3))*args[2].new_tensor(SCALE)
    prior = -.5*((delta-delta.new_tensor([0,0,1.625])).square()/delta.new_tensor(VARIANCE)).sum(-1)
    expected = parent_probabilities(parameters[0]*.2+parameters[1]*prior,null_logit=parameters[2])
    torch.testing.assert_close(model.predict_edges(*args).sigmoid(),expected)
    assert features is images and model._image_motion_calibration==parameters
    assert torch.equal(det[0],images[:,0:1])


def test_calibration_cannot_silently_change_training_or_accept_bad_bounds():
    from image_motion_residual import calibration_parameters
    for bad in ([1.,1.],[2.,1.,-4.],[0.,float('nan'),-4.],[0.,1.,-13.]):
        with pytest.raises(ValueError):
            calibration_parameters(bad)
    model = Neural(); install_motion_residual(model)
    with pytest.raises(ValueError,match='inference'):
        install_image_motion_residual(model,ConstantFlow([0,0,0]),calibration=[0.,1.,-4.5])


def test_optional_zero_weight_shortcut_preserves_scores_and_skips_neural():
    class CountedNeural(Neural):
        def __init__(self): super().__init__(); self.calls=0
        def predict_edges(self,*args):
            self.calls+=1
            return super().predict_edges(*args)
    outputs=[]; models=[]
    for skip in (False,True):
        model=CountedNeural(); install_motion_residual(model)
        install_image_motion_residual(model,ConstantFlow([0,0,1.625]),inference=True,
            calibration=[0.,1.,-4.5],skip_zero_neural=skip)
        model.encode(torch.ones(1,2,3,3,3))
        outputs.append(model.predict_edges(*values()))
        models.append(model)
    torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
    assert models[0].calls==1 and models[1].calls==0
    assert models[0]._image_motion_execution['neural_forward_calls']==1
    assert models[1]._image_motion_execution['zero_weight_skips']==1


@pytest.mark.parametrize('calibration,inference',[(None,True),([.1,1.,-4.5],True),([0.,1.,-4.5],False)])
def test_zero_weight_shortcut_cannot_change_other_policies(calibration,inference):
    model=Neural(); install_motion_residual(model)
    with pytest.raises(ValueError,match='shortcut'):
        install_image_motion_residual(model,ConstantFlow([0,0,0]),inference=inference,
            calibration=calibration,skip_zero_neural=True)
