from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from backward_flow_ops import warp_previous,sample_backward_flow,sparse_backward_loss


def image():
    z,y,x = torch.meshgrid(torch.arange(3.),torch.arange(5.),torch.arange(7.),indexing='ij')
    return (100*z+10*y+x)[None,None]


def test_zero_flow_is_identity_on_noncubic_grid():
    previous = image()
    warped,valid = warp_previous(previous,torch.zeros(1,3,3,5,7),(2.,.5,.25))
    torch.testing.assert_close(warped,previous,atol=1e-4,rtol=1e-6)
    assert valid.all()


@pytest.mark.parametrize('axis,voxel',[(0,2.),(1,.5),(2,.25)])
def test_physical_axis_shift_and_boundary_mask(axis,voxel):
    previous = image()
    flow = torch.zeros(1,3,3,5,7)
    flow[:,axis] = voxel
    warped,valid = warp_previous(previous,flow,(2.,.5,.25))
    output_slice = [slice(None)]*5
    source_slice = [slice(None)]*5
    output_slice[axis+2] = slice(None,-1)
    source_slice[axis+2] = slice(1,None)
    torch.testing.assert_close(warped[tuple(output_slice)],previous[tuple(source_slice)],atol=1e-4,rtol=1e-6)
    assert valid[tuple(output_slice)].all()
    output_slice[axis+2] = -1
    assert not valid[tuple(output_slice)].any()


def test_backward_sign_moves_previous_peak_to_current_position():
    previous = torch.zeros(1,1,3,5,7)
    previous[0,0,1,2,2] = 1
    flow = torch.zeros(1,3,3,5,7)
    flow[:,2] = -.25
    warped,_ = warp_previous(previous,flow,(2.,.5,.25))
    assert warped[0,0,1,2,3] == pytest.approx(1.)
    assert warped[0,0,1,2,2] == pytest.approx(0.,abs=1e-6)


def test_subvoxel_sampling_and_two_daughters_can_share_a_parent():
    z,y,x = torch.meshgrid(torch.arange(3.),torch.arange(5.),torch.arange(7.),indexing='ij')
    flow = torch.stack([2*z,.5*y,.25*(3-x)])[None]
    points = torch.tensor([[[1.,2.,2.],[1.,2.,4.],[.5,1.5,2.5]]])
    sampled,valid = sample_backward_flow(flow,points)
    torch.testing.assert_close(sampled[0,2],torch.tensor([1.,.75,.125]))
    torch.testing.assert_close(points[0,:2,2]+sampled[0,:2,2]/.25,torch.tensor([3.,3.]))
    assert valid.all()


def test_gradients_are_finite_and_outside_points_are_flagged():
    flow = torch.zeros(1,3,3,5,7,requires_grad=True)
    warped,valid = warp_previous(image(),flow,(2.,.5,.25))
    warped[:,:,1,1:4,1:6].mean().backward()
    assert torch.isfinite(flow.grad).all() and flow.grad.abs().sum() > 0
    sampled,valid = sample_backward_flow(flow,torch.tensor([[[1.,2.,3.],[-1.,2.,3.]]]))
    assert valid.tolist() == [[True,False]]
    empty,valid = sample_backward_flow(flow,torch.empty(1,0,3))
    assert empty.shape == (1,0,3) and valid.shape == (1,0)


def test_invalid_voxel_size_or_grid_rejected():
    with pytest.raises(ValueError):
        warp_previous(image(),torch.zeros(1,3,3,5,7),(0.,1.,1.))
    with pytest.raises(ValueError):
        warp_previous(image(),torch.zeros(1,3,4,5,7),(1.,1.,1.))


def test_sparse_loss_both_daughters_and_unknown_has_no_gradient():
    flow = torch.zeros(1,3,3,5,7,requires_grad=True)
    points = torch.tensor([[[1.,2.,2.],[1.,2.,4.],[1.,2.,6.]]])
    targets = torch.tensor([[[0.,0.,.25],[0.,0.,-.25],[float('nan')]*3]])
    loss = sparse_backward_loss(flow,points,targets,torch.tensor([[True,True,False]]))
    assert loss.detach().item() == pytest.approx(.25/3)
    loss.backward()
    assert flow.grad[0,2,1,2,2] < 0
    assert flow.grad[0,2,1,2,4] > 0
    assert flow.grad[0,:,1,2,6].abs().sum() == 0


def test_empty_annotations_have_differentiable_zero_loss():
    flow = torch.zeros(1,3,3,5,7,requires_grad=True)
    loss = sparse_backward_loss(flow,torch.empty(1,0,3),torch.empty(1,0,3),
                               torch.empty(1,0,dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and flow.grad.abs().sum() == 0


def test_observed_outside_or_nonfinite_targets_rejected():
    flow = torch.zeros(1,3,3,5,7)
    mask = torch.tensor([[True]])
    with pytest.raises(ValueError,match='outside'):
        sparse_backward_loss(flow,torch.tensor([[[1.,2.,7.]]]),torch.zeros(1,1,3),mask)
    with pytest.raises(ValueError,match='finite'):
        sparse_backward_loss(flow,torch.tensor([[[1.,2.,3.]]]),torch.full((1,1,3),float('nan')),mask)
