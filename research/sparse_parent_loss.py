"""Parent classification over annotated child links, not all matrix cells.

Only children with an annotated incoming edge are supervised. All detected
parents compete for each such child. A parent may win multiple child columns,
so divisions remain possible. Unannotated child columns carry no target.
This changes the training objective only; inference/scoring stay untouched.
"""


def sparse_parent_loss(logits, target):
    import torch
    if logits.ndim != 2 or logits.shape != target.shape:
        raise ValueError('Matching two-dimensional parent/child matrices required')
    if not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('Binary finite annotation targets required')
    positives = target.sum(dim=0)
    if (positives > 1).any():
        raise ValueError('Multiple annotated parents for one child are invalid')
    active = positives == 1
    if not active.any():
        return logits.sum() * 0.
    selected = logits[:,active]
    if not torch.isfinite(selected).all():
        raise ValueError('Nonfinite supervised association logits')
    log_probs = torch.log_softmax(selected,dim=0)
    return -(log_probs * target[:,active]).sum(dim=0).mean()
