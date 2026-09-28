"""Three physical input scales; backward-compatible underlying architectures."""
import torch
from torch.nn import functional as F
from research.visual_correspondence_models_v1 import VisualCorrespondence, mask_scores


def prepare_patches(stored, coords, valid, augment=False, generator=None):
    b, n = stored.shape[:2]; device=stored.device
    if stored.shape[2:] != (3,15,15,15):
        raise ValueError('Native three-scale storage contract changed')
    coords=coords.clone(); shift=torch.zeros((b,n,3),device=device)
    if augment:
        shift=(torch.rand((b,n,3),device=device,generator=generator)*2.-1.)*.8125
        coords+=shift
    axis=torch.linspace(-5/7,5/7,11,device=device)
    grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).flip(-1)
    scales=torch.tensor([.8125,1.625,3.25],device=device).view(1,3,1,1,1,1)
    shifted=grid+shift.flip(-1).flatten(0,1)[:,None,None,None,None,:]/(7*scales)
    images=F.grid_sample(stored.float().reshape(b*n*3,1,15,15,15),
                        shifted.reshape(b*n*3,11,11,11,3),mode='bilinear',
                        padding_mode='border',align_corners=True).reshape(b,n,3,11,11,11)
    if augment:
        images*=.8+.4*torch.rand((b,1,1,1,1,1),device=device,generator=generator)
        k=int(torch.randint(4,(),device=device,generator=generator).item())
        images=torch.rot90(images,k,(-2,-1))
        coords-=coords[:,:1].clone()
        for _ in range(k):
            old_y=coords[...,1].clone(); coords[...,1]=-coords[...,2]; coords[...,2]=old_y
    return images*valid[:,:,None,None,None,None],coords
