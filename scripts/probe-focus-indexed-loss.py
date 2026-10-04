"""CPU gradient/optimizer/reload functionality; synthetic data only."""
import hashlib
import json
from pathlib import Path
import time


def main():
    import torch
    from focus_indexed_parent_loss import indexed_parent_loss,require_fitting_sample
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20260910)
    scores=torch.tensor([[1.,2.,3.,1.],[0.,1.,2.,0.]],requires_grad=True)
    labels=torch.tensor([0,2,-1,0])
    loss=indexed_parent_loss(scores,labels)
    augmented=torch.cat([scores,torch.full((1,4),-4.5)],dim=0)
    expected=torch.nn.functional.cross_entropy(augmented[:,[0,1,3]].T,torch.tensor([0,2,0]))
    torch.testing.assert_close(loss,expected)
    loss.backward()
    if not ((scores.grad[:,2]==0).all() and (scores.grad[:,1]>0).all()
            and scores.grad[0,0]<0 and scores.grad[0,3]<0):
        raise ValueError('Unknown/null/positive/division-parent gradient semantics changed')
    for invalid in ([-2,0,0,0],[3,0,0,0],[0.,2.,-1.,0.]):
        try:indexed_parent_loss(scores,invalid)
        except ValueError:pass
        else:raise ValueError('Invalid label accepted')
    for sample in ({'role':'diagnostic'},{'role':'embargo'},{}):
        try:require_fitting_sample(sample)
        except ValueError:pass
        else:raise ValueError('Non-fitting sample reached optimizer path')
    require_fitting_sample({'role':'fitting'})
    unknown=torch.ones((2,3),requires_grad=True)
    zero=indexed_parent_loss(unknown,torch.full((3,),-1,dtype=torch.long));zero.backward()
    if zero.item()!=0 or not (unknown.grad==0).all():raise ValueError('Unknown-only columns should carry no loss')
    empty=torch.empty((0,2),requires_grad=True)
    empty_loss=indexed_parent_loss(empty,torch.tensor([0,-1]));empty_loss.backward()
    if empty_loss.item()!=0:raise ValueError('Empty source set has only a null class')
    # A tiny trainable head shows real optimizer updates and exact state recovery.
    head=torch.nn.Linear(5,2)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.03)
    features=torch.randn((8,5));training_labels=torch.tensor([0,1,2,-1,0,1,2,-1])
    before=float(indexed_parent_loss(head(features).T,training_labels).detach())
    for step in range(40):
        optimizer.zero_grad();fit_loss=indexed_parent_loss(head(features).T,training_labels)
        fit_loss.backward();optimizer.step()
    after=float(indexed_parent_loss(head(features).T,training_labels).detach())
    if not after<before:raise ValueError('Synthetic head optimization failed')
    path=Path('/kaggle/working/focus_indexed_loss_cpu.pt')
    torch.save(dict(model=head.state_dict(),optimizer=optimizer.state_dict(),steps=40),path)
    recovered=torch.nn.Linear(5,2);restored=torch.load(path,map_location='cpu',weights_only=True)
    recovered.load_state_dict(restored['model'],strict=True)
    if not torch.equal(head(features),recovered(features)):raise ValueError('Restricted checkpoint reload changed predictions')
    result=dict(status='passed_synthetic_indexed_parent_loss',gpu_used=False,competition_data_read=False,
        matched_existing_objective=True,unknown_logit_gradient_zero=True,known_null_pushes_real_parents_down=True,
        division_daughters_can_share_parent=True,diagnostic_optimizer_guard_passed=True,empty_source_and_unknown_cases_passed=True,
        optimizer_steps=40,initial_nll=before,final_nll=after,strict_checkpoint_reload_exact=True,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        authorized_for_submission=False)
    Path('/kaggle/working/focus_indexed_loss_cpu.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
