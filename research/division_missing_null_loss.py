"""Balance observed division/continuation classes while retaining real nulls."""


def division_missing_null_loss(scores, target, known_null):
    import torch
    from sparse_parent_missing_null import missing_null_targets
    from sparse_parent_loss import sparse_parent_loss
    from sparse_division_balanced_loss import sparse_division_balanced_loss
    augmented, truth = missing_null_targets(scores, target, known_null)
    # Validate all targets before computing either class contribution.
    ordinary = sparse_parent_loss(augmented, truth)
    positive_count = int((target.sum(0) == 1).sum())
    null_mask = truth[-1].bool()
    null_count = int(null_mask.sum())
    if positive_count == 0:
        return ordinary
    positive_loss = sparse_division_balanced_loss(scores, target)
    if null_count == 0:
        return positive_loss
    null_loss = -torch.log_softmax(augmented,dim=0)[-1,null_mask].mean()
    # Null's empirical fraction is unchanged; only positive classes balance.
    fraction = positive_count / (positive_count + null_count)
    return fraction * positive_loss + (1-fraction) * null_loss
