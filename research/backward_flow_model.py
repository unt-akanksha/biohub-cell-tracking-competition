"""Owned image-pair residual 3D U-Net; physical backward flow, no graph edits."""
import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self,source,target):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(source,target,3,padding=1,bias=False),
            nn.GroupNorm(4,target),nn.SiLU(),nn.Conv3d(target,target,3,padding=1,bias=False),
            nn.GroupNorm(4,target))
        self.skip = nn.Identity() if source == target else nn.Conv3d(source,target,1)

    def forward(self,x):
        return F.silu(self.body(x)+self.skip(x))


class BackwardFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([Residual(2,16),Residual(16,32),Residual(32,64),Residual(64,128)])
        self.dec = nn.ModuleList([Residual(192,64),Residual(96,32),Residual(48,16)])
        self.head = nn.Conv3d(16,3,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,pair):
        if pair.ndim != 5 or pair.shape[1] != 2 or min(pair.shape[2:]) < 8:
            raise ValueError('B,2,Z,Y,X previous/current images with dimensions >=8 required')
        skips = []
        x = pair
        for i,block in enumerate(self.enc):
            x = block(F.avg_pool3d(x,2) if i else x)
            skips.append(x)
        for block,skip in zip(self.dec,reversed(skips[:-1])):
            x = block(torch.cat([F.interpolate(x,size=skip.shape[2:],mode='trilinear',
                                               align_corners=False),skip],dim=1))
        return self.head(x)


def observed_targets(coords,masks,transitions,voxel_um):
    """Convert padded GT edges to current-grid locations and parent-child um.

    Voxel size is ALREADY downsampled in official FrameWindowDataset metadata.
    No second downsample factor is applied. Missing/cropped parents are ignored.
    """
    if (coords.ndim != 4 or coords.shape[1] != 2 or coords.shape[-1] != 3
        or masks.shape != coords.shape[:3] or masks.dtype != torch.bool
        or transitions.shape != (coords.shape[0],1,coords.shape[2],coords.shape[2])
        or voxel_um.shape != (coords.shape[0],3)):
        raise ValueError('Expected padded two-frame coordinates, masks, edges and batch voxel sizes')
    if (not torch.isfinite(coords).all() or not torch.isfinite(voxel_um).all()
        or (voxel_um <= 0).any() or not torch.isfinite(transitions).all()
        or ((transitions != 0)&(transitions != 1)).any()):
        raise ValueError('Finite coordinates, positive units and binary edges required')
    edges = transitions[:,0].float()
    if (edges.sum(1)>1).any() or (edges.sum(2)>2).any():
        raise ValueError('No merges and at most two daughters required')
    valid_edges = edges*masks[:,0,:,None]*masks[:,1,None,:]
    annotated = valid_edges.sum(1) == 1
    parents = valid_edges.transpose(1,2)@coords[:,0].float()
    target = (parents-coords[:,1].float())*voxel_um[:,None].float()
    return coords[:,1].float(),target,annotated
