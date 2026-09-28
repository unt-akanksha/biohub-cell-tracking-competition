"""Sparse parent supervision plus hard wrong-child negatives for known parents.

The parent-only objective ignores entirely unannotated child columns. Like
the organizer's annotated-row-or-column mask, this additional term supervises
wrong children only for a parent with a known outgoing annotation. It does
not label all unannotated detections negative or impose a cell-count target.
"""


def sparse_parent_row_loss(scores,target,hard_negatives=4):
    import torch
    from sparse_parent_loss import sparse_parent_loss
    from motion_residual import add_null_target,NULL_LOGIT
    if not isinstance(hard_negatives,int) or hard_negatives < 1:
        raise ValueError('Positive integer hard-negative budget required')
    positive = sparse_parent_loss(*add_null_target(scores,target))
    rows = torch.nonzero(target.sum(1) > 0).flatten()
    if not len(rows):
        return positive
    null = torch.full_like(scores[:1],NULL_LOGIT)
    log_total = torch.logsumexp(torch.cat([scores,null]),dim=0)
    penalties = []
    for row in rows:
        wrong = target[row] == 0
        if not wrong.any():
            continue
        # Stable -log(1-P(parent|child)); avoids saturation from clamping P.
        alternative = scores.clone()
        alternative[row] = -torch.inf
        log_alternative = torch.logsumexp(torch.cat([alternative,null]),dim=0)
        nll = (log_total-log_alternative)[wrong]
        penalties.append(nll.topk(min(hard_negatives,len(nll))).values.mean())
    return positive+(torch.stack(penalties).mean() if penalties else scores.sum()*0.)
