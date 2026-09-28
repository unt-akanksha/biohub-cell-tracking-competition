"""Append a true null target only for externally verified missing parents."""


def missing_null_targets(scores, target, known_null, null_logit=-4.5):
    import torch
    if scores.ndim != 2 or scores.shape != target.shape:
        raise ValueError('Matching parent-child matrices required')
    mask = torch.as_tensor(known_null, dtype=torch.bool, device=scores.device)
    if mask.shape != (scores.shape[1],) or ((target.sum(0) > 0) & mask).any():
        raise ValueError('Null mask must not contradict an annotated parent')
    logits = torch.cat([scores, torch.full((1, scores.shape[1]), null_logit,
                       dtype=scores.dtype, device=scores.device)], dim=0)
    truth = torch.cat([target, mask.to(target.dtype).unsqueeze(0)], dim=0)
    return logits, truth
