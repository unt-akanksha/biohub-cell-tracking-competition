from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from backward_flow_image_loss import image_alignment,physical_smoothness,boundary_distance


def images():
    generator=torch.Generator().manual_seed(21)
    return torch.rand(1,1,5,7,9,generator=generator)


def test_identity_and_constant_background_have_zero_loss():
    current=images(); flow=torch.zeros(1,3,5,7,9,requires_grad=True)
    result=image_alignment(current,current,flow,(2.,.5,.25))
    assert result['loss'].item()==pytest.approx(0.,abs=1e-6)
    result['loss'].backward()
    assert torch.isfinite(flow.grad).all()
    background=torch.zeros_like(current)
    result=image_alignment(background,background,flow,(2.,.5,.25))
    assert result['texture_patches']==0 and result['loss'].item()==0


def test_correct_backward_translation_beats_zero_and_wrong_sign():
    previous=torch.zeros(1,1,5,7,9); previous[0,0,2,3,3]=1.
    current=torch.zeros_like(previous); current[0,0,2,3,4]=1.
    zero=torch.zeros(1,3,5,7,9)
    correct=zero.clone(); correct[:,2]=-.25
    wrong=-correct
    # Boundary distance is reported separately: compare correspondence on the
    # textured nucleus support, which lies safely away from image boundaries.
    good=image_alignment(previous,current,correct,(2.,.5,.25))
    assert good['ssim_loss'].item()==pytest.approx(0.,abs=1e-6)
    assert good['ssim_loss']<image_alignment(previous,current,zero,(2.,.5,.25))['ssim_loss']
    assert good['ssim_loss']<image_alignment(previous,current,wrong,(2.,.5,.25))['ssim_loss']


def test_out_of_volume_cannot_remove_difficult_image_patches():
    current=images(); flow=torch.full((1,3,5,7,9),100.,requires_grad=True)
    result=image_alignment(current,current,flow,(1.,1.,1.))
    assert result['invalid_texture_patches']==result['texture_patches']>0
    assert result['ssim_loss'].item()==1 and result['loss'].item()>1
    result['loss'].backward()
    assert torch.isfinite(flow.grad).all() and (flow.grad>0).all()


def test_physical_smoothness_is_spacing_aware_and_constant_flow_free():
    flow=torch.ones(1,3,5,7,9)
    assert physical_smoothness(flow,(2.,.5,.25)).item()==0
    z,y,x=torch.meshgrid(torch.arange(5.),torch.arange(7.),torch.arange(9.),indexing='ij')
    field=torch.stack([.1*z*2.,.1*y*.5,.1*x*.25])[None]
    assert physical_smoothness(field,(2.,.5,.25)).item()==pytest.approx(.1/3,abs=1e-7)


def test_nontrivial_image_loss_has_finite_flow_gradient():
    previous=images(); current=previous.roll(1,-1)
    flow=torch.full((1,3,5,7,9),.1,requires_grad=True)
    result=image_alignment(previous,current,flow,(1.,1.,1.))
    (result['loss']+.01*physical_smoothness(flow,(1.,1.,1.))).backward()
    assert torch.isfinite(flow.grad).all() and flow.grad.abs().sum()>0


def test_invalid_shapes_intensities_or_voxel_sizes_rejected():
    current=images(); flow=torch.zeros(1,3,5,7,9)
    with pytest.raises(ValueError): image_alignment(current[:,:,:2],current,flow,(1.,1.,1.))
    with pytest.raises(ValueError): image_alignment(current*float('nan'),current,flow,(1.,1.,1.))
    with pytest.raises(ValueError): boundary_distance(flow,(0.,1.,1.))
