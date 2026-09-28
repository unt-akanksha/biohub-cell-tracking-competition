"""Owned physical-unit backward-flow primitives; no tracking/scorer changes.

Flow channels are Z,Y,X microns at CURRENT-frame grid points. Reconstructing
the current image samples the PREVIOUS image at current_grid + flow/voxel_um.
Point coordinates are in the downsampled image grid, not original voxels.
"""
import torch
import torch.nn.functional as F


def _shape(flow):
    if flow.ndim != 5 or flow.shape[1] != 3 or min(flow.shape[2:]) < 2:
        raise ValueError('B,3,Z,Y,X flow with spatial dimensions at least two required')
    if not torch.isfinite(flow).all():
        raise ValueError('Finite physical flow required')
    return flow.shape[2:]


def _normalized_xyz(coords,shape):
    extent = coords.new_tensor(shape)-1
    normalized = 2*coords/extent-1
    return normalized[..., [2,1,0]]


def warp_previous(previous,backward_um,voxel_um):
    shape = _shape(backward_um)
    if previous.shape != (backward_um.shape[0],1,*shape) or previous.device != backward_um.device:
        raise ValueError('Previous image and flow must share batch/grid/device')
    voxel = torch.as_tensor(voxel_um,dtype=torch.float32,device=backward_um.device)
    if voxel.shape != (3,) or not torch.isfinite(voxel).all() or (voxel <= 0).any():
        raise ValueError('Three finite positive physical voxel sizes required')
    axes = [torch.arange(n,dtype=torch.float32,device=backward_um.device) for n in shape]
    grid = torch.stack(torch.meshgrid(*axes,indexing='ij'),dim=-1)
    source = grid.unsqueeze(0)+backward_um.float().permute(0,2,3,4,1)/voxel
    valid = ((source >= 0)&(source <= source.new_tensor(shape)-1)).all(-1).unsqueeze(1)
    warped = F.grid_sample(previous.float(),_normalized_xyz(source,shape),mode='bilinear',
                           padding_mode='zeros',align_corners=True)
    return warped,valid


def sample_backward_flow(backward_um,points_zyx):
    shape = _shape(backward_um)
    if (points_zyx.ndim != 3 or points_zyx.shape[0] != backward_um.shape[0]
        or points_zyx.shape[-1] != 3 or points_zyx.device != backward_um.device
        or not torch.isfinite(points_zyx).all()):
        raise ValueError('Finite B,N,3 current-frame grid points required')
    points = points_zyx.float()
    valid = ((points >= 0)&(points <= points.new_tensor(shape)-1)).all(-1)
    if points.shape[1] == 0:
        return backward_um.new_empty((backward_um.shape[0],0,3)),valid
    grid = _normalized_xyz(points,shape).unsqueeze(2).unsqueeze(2)
    sampled = F.grid_sample(backward_um.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
    return sampled[:,:, :,0,0].transpose(1,2),valid


def sparse_backward_loss(backward_um,points_zyx,target_um,annotated):
    """Coordinate-mean L1 on observed incoming links; unknown cells add no loss.

    Targets are parent-minus-child physical displacements. Both daughters
    remain separate observations; births and unknown parents are masked out.
    Padding targets may be nonfinite ONLY where annotated is false.
    """
    if (target_um.shape != points_zyx.shape or target_um.device != backward_um.device
        or annotated.shape != points_zyx.shape[:2] or annotated.dtype != torch.bool
        or annotated.device != backward_um.device):
        raise ValueError('Aligned physical targets and boolean annotation mask required')
    sampled,valid = sample_backward_flow(backward_um,points_zyx)
    if (annotated & ~valid).any():
        raise ValueError('Annotated child lies outside the current image grid')
    if not torch.isfinite(target_um[annotated]).all():
        raise ValueError('Observed displacements must be finite')
    if not annotated.any():
        return backward_um.sum()*0
    return (sampled[annotated]-target_um[annotated].float()).abs().mean()
