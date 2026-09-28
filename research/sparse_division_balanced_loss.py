"""Balance annotated continuation and two-daughter parent classification.

Only child columns with an annotated incoming edge are supervised. A division
requires two explicitly matched annotated daughters of the same parent; an
unmatched/unannotated detection is never turned into a negative label. This
is a training-loss hypothesis, not an inference policy or promoted model.
"""


def sparse_division_balanced_loss(scores, target):
    import torch
    from motion_residual import add_null_target
    from sparse_parent_loss import sparse_parent_loss

    augmented, truth = add_null_target(scores, target)
    # Preserve the existing shape/finite/binary/multiple-parent checks.
    ordinary = sparse_parent_loss(augmented, truth)
    active = target.sum(0) == 1
    if not active.any():
        return ordinary
    child_count = target.sum(1)
    if (child_count > 2).any():
        raise ValueError('Annotated parent has more than two children')
    division_columns = ((target > 0) & (child_count == 2).unsqueeze(1)).any(0) & active
    continuation_columns = active & ~division_columns
    if not division_columns.any() or not continuation_columns.any():
        return ordinary
    nll = -(torch.log_softmax(augmented, dim=0) * truth).sum(0)
    return .5 * (nll[division_columns].mean() + nll[continuation_columns].mean())
