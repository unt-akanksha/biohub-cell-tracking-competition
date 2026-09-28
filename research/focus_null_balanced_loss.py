"""Fitting-count-derived weighting for verified nulls; unknowns stay ignored."""
import math


def fitting_null_weight(known_parent,known_absent):
    if (type(known_parent)!=int or type(known_absent)!=int or known_parent<=0 or known_absent<=0):
        raise ValueError('Positive integer fitting-only label counts required')
    return math.sqrt(known_parent/known_absent)


def null_balanced_parent_loss(scores,labels,null_weight):
    import torch
    if not math.isfinite(null_weight) or null_weight<=0:raise ValueError('Finite positive fixed null weight required')
    if scores.ndim!=2 or not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError('Finite source-by-target logits required')
    labels=torch.as_tensor(labels,device=scores.device);ns,nt=scores.shape
    if (labels.shape!=(nt,) or labels.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64)
        or (labels< -1).any() or (labels>ns).any()):raise ValueError('Exact parent/null/unknown integer labels required')
    active=labels>=0
    if not active.any():return scores.sum()*0.
    logits=torch.cat([scores,torch.full((1,nt),-4.5,dtype=scores.dtype,device=scores.device)],dim=0)
    selected=labels[active].long();logp=torch.log_softmax(logits[:,active],dim=0)
    losses=-logp.gather(0,selected.unsqueeze(0)).squeeze(0)
    weights=torch.where(selected==ns,logp.new_tensor(null_weight),logp.new_tensor(1.))
    return (losses*weights).sum()/weights.sum()
