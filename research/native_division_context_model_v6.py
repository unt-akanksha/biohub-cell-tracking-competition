"""Trainable native encoder with chronological, daughter-symmetric event fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from research.visual_correspondence_models_v1 import VisualCorrespondence
from research.native_correspondence_models_v2 import prepare_patches


def paired_crops(original,context,coords,context_valid,augment=False,generator=None):
    valid=torch.ones_like(context_valid)
    state=generator.get_state() if augment else None
    primary,transformed=prepare_patches(original,coords,valid,augment=augment,generator=generator)
    if augment:generator.set_state(state)
    extra,other_coords=prepare_patches(context,coords,context_valid,augment=augment,generator=generator)
    if not torch.equal(transformed,other_coords):raise ValueError('Temporal crops lost coordinate consistency')
    return primary,extra,transformed


class TemporalDivision(nn.Module):
    def __init__(self,origin_family='resnet3d'):
        super().__init__()
        self.encoder=VisualCorrespondence(origin_family,input_channels=3).encoder
        self.temporal=nn.Sequential(nn.Linear(1025,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,256),nn.GELU())
        self.event=nn.Sequential(nn.Linear(1286,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.event[-1].weight);nn.init.zeros_(self.event[-1].bias)

    def load_source_encoder(self,state):
        selected={k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(selected,strict=True)

    def forward(self,original,context,coords,context_valid,use_context=True):
        if original.shape!=context.shape or original.shape[1:]!=(3,3,11,11,11):raise ValueError('Expected three node crops')
        b=original.shape[0]
        current=self.encoder(original.flatten(0,1)).reshape(b,3,256)
        valid=context_valid if use_context else torch.zeros_like(context_valid)
        future=current
        if bool(valid.any()):
            encoded=self.encoder(context.flatten(0,1)[valid.flatten()])
            future=current.flatten(0,1).index_copy(0,valid.flatten().nonzero().flatten(),encoded).reshape(b,3,256)
        current=F.normalize(current.float(),dim=-1);future=F.normalize(future.float(),dim=-1)
        parent=torch.tensor([True,False,False],device=original.device)[None,:,None]
        before=torch.where(parent,future,current);after=torch.where(parent,current,future)
        # Invalid context falls back to current embeddings with an explicit mask;
        # the model receives zero temporal change, not a fabricated observation.
        temporal_input=torch.cat((before,after,after-before,before*after,valid[:,:,None].float()),dim=-1)
        tokens=self.temporal(temporal_input)
        p,a,c=tokens[:,0],tokens[:,1],tokens[:,2];mean=(a+c)*.5
        representation=torch.cat((p,mean,(a-c).abs(),p*mean,a*c),dim=-1)
        delta=(coords[:,1:]-coords[:,:1]).float();r=delta.square().sum(-1).sqrt();ordered=r.sort(-1).values
        separation=(coords[:,1]-coords[:,2]).square().sum(-1).sqrt()
        midpoint=delta.mean(1).square().sum(-1).sqrt()
        cosine=(delta[:,0]*delta[:,1]).sum(-1)/(r[:,0]*r[:,1]).clamp_min(1e-6)
        geometry=torch.stack((ordered[:,0]/10,ordered[:,1]/10,separation/10,midpoint/10,cosine,(ordered[:,1]-ordered[:,0])/10),dim=-1)
        return self.event(torch.cat((representation,geometry),dim=-1)).squeeze(-1).float()
