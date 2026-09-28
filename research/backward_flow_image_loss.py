"""Owned local 3D image alignment and physical smoothness loss primitives.

No labels or graph metric enter these terms. They are prospective components,
not an evaluated training recipe. Backward warping permits daughter convergence.
"""
import torch
import torch.nn.functional as F
from backward_flow_ops import warp_previous,_shape


def _voxel(flow,voxel_um):
    voxel=torch.as_tensor(voxel_um,device=flow.device,dtype=torch.float32)
    if voxel.shape!=(3,) or not torch.isfinite(voxel).all() or (voxel<=0).any():
        raise ValueError('Three finite positive physical voxel sizes required')
    return voxel


def boundary_distance(backward_um,voxel_um):
    """Differentiable outside-grid distance; warping outside cannot evade loss."""
    shape=_shape(backward_um); voxel=_voxel(backward_um,voxel_um)
    axes=[torch.arange(n,device=backward_um.device,dtype=torch.float32) for n in shape]
    grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),dim=0).unsqueeze(0)
    source=grid+backward_um.float()/voxel[None,:,None,None,None]
    extent=source.new_tensor(shape)[None,:,None,None,None]-1
    return (F.relu(-source)+F.relu(source-extent)).mean()


def physical_smoothness(backward_um,voxel_um):
    """Mean absolute physical displacement derivative, not an invertibility rule."""
    _shape(backward_um); voxel=_voxel(backward_um,voxel_um)
    return torch.stack([(torch.diff(backward_um.float(),dim=axis+2)/voxel[axis]).abs().mean()
                        for axis in range(3)]).mean()


def image_alignment(previous,current,backward_um,voxel_um):
    """3x3x3 SSIM on textured current-image patches with explicit boundaries.

    Texture support depends only on the observed current image, never on the
    prediction. Invalid sampling receives maximal patch dissimilarity plus a
    differentiable boundary penalty, not a disappearing masked objective.
    Input intensity follows the existing per-frame quantile normalization.
    """
    shape=_shape(backward_um)
    if (min(shape)<3 or previous.shape!=current.shape or previous.shape!=(backward_um.shape[0],1,*shape)
        or previous.device!=current.device or previous.device!=backward_um.device
        or not torch.isfinite(previous).all() or not torch.isfinite(current).all()):
        raise ValueError('Finite aligned B,1,Z,Y,X images, each spatial dimension >=3 required')
    warped,valid=warp_previous(previous,backward_um,voxel_um)
    current=current.float()
    pool=lambda value:F.avg_pool3d(value,3,stride=1)
    ma,mb=pool(warped),pool(current)
    va=(pool(warped.square())-ma.square()).clamp_min(0)
    vb=(pool(current.square())-mb.square()).clamp_min(0)
    cov=pool(warped*current)-ma*mb
    ssim=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma.square()+mb.square()+.01**2)*(va+vb+.03**2))
    support=(vb.detach()>1e-6)
    valid_patch=pool(valid.float())>=1.-1e-6
    patch_loss=torch.where(valid_patch,(1.-ssim.clamp(-1.,1.))/2.,torch.ones_like(ssim))
    similarity=patch_loss[support].mean() if support.any() else backward_um.sum()*0
    outside=boundary_distance(backward_um,voxel_um)
    return dict(loss=similarity+outside,ssim_loss=similarity,boundary_loss=outside,
        texture_patches=int(support.sum()),invalid_texture_patches=int((support&~valid_patch).sum()))
