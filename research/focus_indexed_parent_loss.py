"""Audited integer labels -> existing sparse-parent/null training objective."""


def indexed_parent_loss(scores,labels):
    import torch
    try:
        from sparse_parent_missing_null import missing_null_targets
        from sparse_parent_loss import sparse_parent_loss
    except ModuleNotFoundError:
        from research.sparse_parent_missing_null import missing_null_targets
        from research.sparse_parent_loss import sparse_parent_loss
    if scores.ndim!=2 or not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError('Finite floating source-by-target logits required')
    labels=torch.as_tensor(labels,device=scores.device)
    ns,nt=scores.shape
    if (labels.shape!=(nt,) or labels.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64)
        or (labels< -1).any() or (labels>ns).any()):
        raise ValueError('Integer labels: -1 unknown, source row or ns for known null')
    target=torch.zeros_like(scores)
    present=(labels>=0)&(labels<ns)
    columns=torch.nonzero(present,as_tuple=False).flatten()
    target[labels[columns].long(),columns]=1.
    augmented,truth=missing_null_targets(scores,target,labels==ns)
    return sparse_parent_loss(augmented,truth)


def require_fitting_sample(sample):
    """Diagnostic examples must never enter the optimizer input path."""
    if sample.get('role')!='fitting':
        raise ValueError('Only fitting-role cached samples may enter optimizer steps')
    return sample
