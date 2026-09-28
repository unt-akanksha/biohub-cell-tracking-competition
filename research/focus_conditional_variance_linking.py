"""Label-free target-dependent variance; unchanged complete raw-node graph policy."""
import numpy as np

from research.focus_conditional_motion import predict as mean_predict
from research.focus_conditional_variance import predict as variance_predict
from research.independent_motion_prior import SCALE, NULL_LOGIT


def link(coords, flow, model):
    points = np.asarray(coords, float)
    motion = np.asarray(flow, float)
    if (points.ndim != 2 or points.shape[1] != 4 or motion.shape != (len(points), 3)
            or not np.isfinite(points).all() or not np.isfinite(motion).all()
            or (points < 0).any() or np.any(points[:, 0] != np.floor(points[:, 0]))):
        raise ValueError('Finite unchanged native geometry and aligned flow required')
    x = np.column_stack([motion, points[:, 1:]])
    correction = mean_predict(model['mean_model'], x)
    corrected = motion + correction
    corrected[points[:, 0] == 0] = 0.
    variance = variance_predict(model, x)
    if not np.isfinite(corrected).all():
        raise ValueError('Finite corrected motion required')
    edges = []
    for t in sorted(set(points[:, 0].astype(int))):
        src, tgt = np.flatnonzero(points[:, 0] == t), np.flatnonzero(points[:, 0] == t + 1)
        if max(len(src), len(tgt)) > 2048:
            raise ValueError('Guard exceeded; no node truncation allowed')
        if not len(src) or not len(tgt):
            continue
        delta = points[src, None, 1:] * SCALE - (points[None, tgt, 1:] * SCALE + corrected[None, tgt])
        weights = np.exp(-.5 * np.sum(delta ** 2 / variance[None, tgt], axis=-1))
        probabilities = weights / (weights.sum(axis=0, keepdims=True) + np.exp(NULL_LOGIT))
        proposals = [(float(probabilities[i, j]), int(src[i]), int(tgt[j]))
                     for i, j in zip(*np.where(probabilities > .5))]
        degree, assigned = {}, set()
        for p, s, d in sorted(proposals, key=lambda r: (-r[0], r[1], r[2])):
            if d not in assigned and degree.get(s, 0) < 2:
                edges.append((s, d, p))
                assigned.add(d)
                degree[s] = degree.get(s, 0) + 1
    return edges
