"""Training-only case-balanced sampling of annotated division windows."""
import numpy as np


def division_flags(targets):
    flags = []
    for values in targets:
        target = np.asarray(values)
        if (target.ndim != 3 or not np.isfinite(target).all()
                or not ((target == 0) | (target == 1)).all()
                or np.any(target.sum(-1) > 2) or np.any(target.sum(-2) > 1)):
            raise ValueError('Binary temporal targets with no merges and at most two daughters required')
        flags.append(bool(np.any(target.sum(-1) == 2)))
    return np.asarray(flags, dtype=bool)


def balanced_weights(flags):
    flags = np.asarray(flags)
    if flags.ndim != 1 or flags.dtype != np.dtype(bool):
        raise ValueError('Boolean per-window labels required')
    positive = int(flags.sum())
    negative = len(flags) - positive
    if positive == 0 or negative == 0:
        raise ValueError('Both division and ordinary training windows required')
    # A frozen 50/50 case sampler, not a probability calibration or an inference prior.
    weights = np.where(flags, .5 / positive, .5 / negative)
    return weights
