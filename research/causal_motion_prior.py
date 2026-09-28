"""Training-fitted causal motion; exact same nodes, no truth/history oracle."""
import numpy as np

from research.independent_motion_prior import SCALE, VARIANCE, NULL_LOGIT


def link_causal_motion(coords, model):
    points = np.asarray(coords, dtype=float).reshape(-1, 4)
    if (not np.isfinite(points).all() or np.any(points < 0)
            or np.any(points[:, 0] != np.floor(points[:, 0]))):
        raise ValueError('Finite nonnegative coordinates with integer times required')
    alpha = np.asarray(model['alpha'], dtype=float)
    intercept = np.asarray(model['intercept_um'], dtype=float)
    residual = np.asarray(model['residual_variance_um2'], dtype=float)
    if (any(x.shape != (3,) for x in (alpha, intercept, residual))
            or not all(np.isfinite(x).all() for x in (alpha, intercept, residual))
            or np.any(alpha < 0) or np.any(alpha > 1) or np.any(residual <= 0)):
        raise ValueError('Finite conservative axiswise model required')
    # Quantized x(t+1) - (1+a)x(t) + a*x(t-1), independent uniform errors.
    forecast_variance = residual + 1.625**2 / 6 * (1 + alpha + alpha**2)
    edges, incoming, outdegree = [], {}, {}
    for time in sorted(set(points[:, 0].astype(int))):
        src = np.flatnonzero(points[:, 0] == time)
        tgt = np.flatnonzero(points[:, 0] == time + 1)
        if not len(src) or not len(tgt):
            continue
        if max(len(src), len(tgt)) > 2048:
            raise ValueError('Diagnostic memory guard exceeded; no truncation')
        prediction = points[src, 1:] * SCALE
        variance = np.tile(VARIANCE, (len(src), 1))
        for i, source in enumerate(src):
            parent = incoming.get(int(source))
            # Only our already accepted, unbranched past edge is usable.
            if parent is not None and outdegree[parent] == 1:
                velocity = (points[source, 1:] - points[parent, 1:]) * SCALE
                prediction[i] += velocity * alpha + intercept
                variance[i] = forecast_variance
        delta = prediction[:, None, :] - points[None, tgt, 1:] * SCALE
        logits = -.5 * np.sum(delta**2 / variance[:, None, :], axis=-1)
        weights = np.exp(logits)
        probs = weights / (weights.sum(axis=0, keepdims=True) + np.exp(NULL_LOGIT))
        proposals = [(float(probs[i, j]), int(src[i]), int(tgt[j]))
                     for i, j in zip(*np.where(probs > .5))]
        for probability, source, target in sorted(proposals, key=lambda r: (-r[0], r[1], r[2])):
            if target not in incoming and outdegree.get(source, 0) < 2:
                edges.append((source, target, probability))
                incoming[target] = source
                outdegree[source] = outdegree.get(source, 0) + 1
    return edges
