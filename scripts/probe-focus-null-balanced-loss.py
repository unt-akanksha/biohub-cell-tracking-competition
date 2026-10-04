"""Tiny CPU Torch test of weighted supervision; never initializes CUDA."""
import json
from pathlib import Path
import time


def main():
    started=time.monotonic();root=Path(__file__).resolve().parent;target=root/'result.json'
    if target.exists():raise ValueError('Do not overwrite CPU loss smoke')
    import torch
    from focus_null_balanced_loss import fitting_null_weight,null_balanced_parent_loss
    from focus_indexed_parent_loss import indexed_parent_loss
    torch.set_num_threads(1)
    scores=torch.tensor([[.2,.9,-.2,.1],[.7,.1,.4,-.3]],dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([0,1,2,-1]);weight=fitting_null_weight(10754,161)
    original=indexed_parent_loss(scores,labels);equal=null_balanced_parent_loss(scores,labels,1.)
    torch.testing.assert_close(equal,original,rtol=0,atol=1e-12)
    loss=null_balanced_parent_loss(scores,labels,weight);loss.backward()
    if not torch.isfinite(scores.grad).all() or (scores.grad[:,3]!=0).any() or not (scores.grad[:,2]>0).all():
        raise ValueError('Unknown gradient or known-null gradient direction failed')
    unknown=torch.zeros((2,2),dtype=torch.float64,requires_grad=True)
    null_balanced_parent_loss(unknown,torch.full((2,),-1),weight).backward()
    if (unknown.grad!=0).any():raise ValueError('All-unknown batch has gradient')
    divisions=torch.zeros((2,2),dtype=torch.float64,requires_grad=True)
    null_balanced_parent_loss(divisions,torch.tensor([1,1]),weight).backward()
    if not (divisions.grad[1]<0).all():raise ValueError('Same-parent daughters not independently supervised')
    try:null_balanced_parent_loss(scores,labels.double(),weight)
    except ValueError:pass
    else:raise ValueError('Floating labels were accepted')
    empty=torch.empty((0,2),dtype=torch.float64,requires_grad=True)
    empty_loss=null_balanced_parent_loss(empty,torch.zeros(2,dtype=torch.int64),weight);empty_loss.backward()
    if float(empty_loss)!=0:raise ValueError('Empty-source deterministic null is not zero loss')
    parameter=torch.nn.Parameter(torch.zeros((2,3),dtype=torch.float64));known=torch.tensor([0,1,2])
    optimizer=torch.optim.SGD([parameter],lr=.5);initial=float(null_balanced_parent_loss(parameter,known,weight).detach())
    for _ in range(40):
        optimizer.zero_grad();value=null_balanced_parent_loss(parameter,known,weight);value.backward();optimizer.step()
    final=float(null_balanced_parent_loss(parameter,known,weight).detach())
    if not final<initial or torch.cuda.is_initialized():raise ValueError('CPU-only synthetic learning check failed')
    result=dict(status='passed_cpu_null_weighted_loss_smoke',null_weight=weight,unit_weight_loss_delta=float(abs(equal-original).detach()),
        unknown_gradient_exact_zero=True,known_null_gradient_direction=True,same_parent_daughters_supported=True,
        integer_labels_guarded=True,empty_source_null_loss=0.,synthetic_initial=initial,synthetic_final=final,
        torch_version=torch.__version__,cuda_initialized=False,real_training_performed=False,authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
