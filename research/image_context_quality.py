"""Tie-aware source calibration and fixed-threshold diagnostic metrics."""
import numpy as np


def metrics(labels, scores, threshold=None):
    y = np.asarray(labels) > .5
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError('Finite aligned vectors required')
    if not y.any() or y.all():
        raise ValueError('Both classes required')
    order = np.argsort(-s, kind='stable')
    ranked, values = y[order], s[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values)-1]
    tp = np.cumsum(ranked)[ends]
    precision = tp / (ends+1)
    ap = float(np.sum(np.diff(np.r_[0, tp]) * precision) / y.sum())
    safe = s[y & (s > s[~y].max())]
    source_threshold = float(safe.min()) if len(safe) >= 2 else None
    result = dict(rows=len(y), positives=int(y.sum()), average_precision=ap,
                  zero_fp_tp=int(len(safe)), source_threshold=source_threshold,
                  binary_cross_entropy=float(np.mean(np.logaddexp(0., s) - y*s)))
    result['source_gate_passed'] = ap >= .55 and source_threshold is not None
    if threshold is not None:
        pred = s >= threshold
        a = int((pred & y).sum()); b = int((pred & ~y).sum()); c = int((~pred & y).sum())
        result['fixed_threshold'] = dict(threshold=float(threshold), tp=a, fp=b, fn=c,
                                        jaccard=a / (a+b+c))
    return result


def utility(result):
    return (result['zero_fp_tp'], result['average_precision'], -result['binary_cross_entropy'])
