"""Diagnostic upper bound for a sum-of-numerators / sum-of-denominators selector.

No trained router, prediction graph, identity feature, or routing rule is exported.
"""
import numpy as np


def fractional_bound(numerator, denominator):
    n, d = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    if (n.ndim != 2 or n.shape != d.shape or min(n.shape, default=0) < 1
            or not np.isfinite(n).all() or not np.isfinite(d).all()
            or (n < 0).any() or (d <= 0).any()):
        raise ValueError('Finite nonnegative numerators and positive denominators required')
    ratio = 0.
    rows = np.arange(len(n))
    for iteration in range(1, 101):
        choices = np.argmax(n - ratio * d, axis=1)
        new_ratio = float(n[rows, choices].sum() / d[rows, choices].sum())
        if abs(new_ratio - ratio) <= 1e-14:
            gap = float(np.max(n - new_ratio * d, axis=1).sum())
            if abs(gap) > 1e-9 * max(1., float(n[rows, choices].sum())):
                raise ValueError('Fractional optimum certificate failed')
            return dict(ratio=new_ratio, iterations=iteration, residual=gap,
                        aggregate_arm_counts=np.bincount(choices, minlength=n.shape[1]).tolist())
        ratio = new_ratio
    raise ValueError('Fractional bound did not converge')
