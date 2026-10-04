"""CPU-only model and native-scale contract checks; safe alongside image reads."""
import argparse
import json
from pathlib import Path
import sys
import torch

parser=argparse.ArgumentParser(); parser.add_argument('--module-root',type=Path,required=True)
args=parser.parse_args(); sys.path.insert(0,str(args.module_root))
from research.native_correspondence_models_v2 import VisualCorrespondence,prepare_patches,mask_scores

torch.set_num_threads(2); torch.manual_seed(53)
stored=torch.randn(2,4,3,15,15,15); coord=torch.randn(2,4,3); valid=torch.ones(2,4,dtype=torch.bool)
x,c=prepare_patches(stored,coord,valid)
assert torch.allclose(x,stored[...,2:13,2:13,2:13],atol=1e-5)
assert torch.equal(c,coord)
generator=torch.Generator().manual_seed(54)
xa,ca=prepare_patches(stored,coord,valid,True,generator)
assert xa.shape==x.shape and torch.isfinite(xa).all() and torch.isfinite(ca).all()
assert torch.equal(ca[:,:1],torch.zeros_like(ca[:,:1]))
assert torch.equal(mask_scores(torch.ones(2,4),torch.zeros(2,3,dtype=torch.bool))[:,-1],torch.ones(2))
parameters={}
for family in ('resnet3d','token_transformer3d'):
    model=VisualCorrespondence(family,input_channels=3).eval()
    parameters[family]=sum(p.numel() for p in model.parameters())
    y=model(x,c,valid)
    prior=torch.cat((-2*((c[:,1:]-c[:,:1])/10).square().sum(-1),torch.full((2,1),-8.)),dim=-1)
    assert torch.allclose(y,prior,atol=1e-6)
    torch.nn.init.normal_(model.edge_head[-1].weight,std=.01)
    y=model(x,c,valid); order=torch.tensor([0,3,1,2]); output_order=torch.tensor([2,0,1,3])
    assert torch.allclose(model(x[:,order],c[:,order],valid[:,order]),y[:,output_order],atol=2e-5)
    empty=valid.clone(); empty[:,1:]=False
    ye=model(x,c,empty)
    assert torch.isfinite(ye).all() and torch.equal(ye.argmax(-1),torch.tensor([3,3]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    optimizer.zero_grad(); torch.nn.functional.cross_entropy(y,torch.tensor([0,1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.encoder.parameters())
    optimizer.step()
print(json.dumps(dict(status='passed',parameters=parameters,checks=['native_center_crop','physical_augmentation',
    'null_mask','zero_residual_prior','candidate_permutation','empty_parent_set','finite_backward','visual_gradient'])))
