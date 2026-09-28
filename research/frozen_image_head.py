"""Small source-fitted classifier on fixed microscopy/image-context features."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

BLOCKS = (1283, 18, 45)
PENALTIES = (.1, .01, .001, .0001)


def morphology(context, mask):
    c = np.asarray(context, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if c.ndim != 3 or c.shape[1:] != (43, 8) or m.shape != c.shape[:2]:
        raise ValueError('Invalid image-context layout')
    if not np.isfinite(c).all() or not m[:, :3].all() or m[:, 27:].any():
        raise ValueError('Finite image-only context with three anchors required')
    rows = []
    for tokens, valid in zip(c, m):
        anchored = np.zeros((3, 3, 4), dtype=np.float64)
        global_frames = []
        for t in range(3):
            peaks = tokens[3+t*8:11+t*8][valid[3+t*8:11+t*8]]
            global_frames.extend((len(peaks)/8., peaks[:, 4].mean() if len(peaks) else 0.,
                                  peaks[:, 7].mean()/4. if len(peaks) else 0.))
            for a in range(3):
                if not len(peaks):
                    anchored[a, t] = (1., 0., 0., 0.)
                    continue
                distances = np.linalg.norm(peaks[:, 1:4] - tokens[a, 1:4], axis=1)
                # Deterministic tie group average avoids peak-order dependence.
                nearest = distances == distances.min()
                anchored[a, t] = (min(float(distances.min()), 1.),
                    peaks[nearest, 4].mean(), peaks[nearest, 7].mean()/4., 1.)
        anchored = anchored.reshape(3, 12)
        rows.append(np.r_[anchored[0], anchored[1:].mean(axis=0),
                          np.abs(anchored[1]-anchored[2]), global_frames])
    return np.asarray(rows, dtype=np.float64).reshape(len(c), 45)


def stratum_weights(labels, eligible, quality):
    y = np.asarray(labels) > .5
    e = np.asarray(eligible, dtype=bool)
    q = np.asarray(quality, dtype=np.float64)
    if not (y.shape == e.shape == q.shape) or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError('Aligned positive quality weights required')
    if not y.any() or y.all():
        raise ValueError('Both optimization classes required')
    weights = np.zeros_like(q)
    for label in (False, True):
        for gate in (False, True):
            select = (y == label) & (e == gate)
            if select.any():
                weights[select] = q[select] / q[select].sum()
    return weights / weights.sum()


def standardizer(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != sum(BLOCKS) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite fixed-width optimization features required')
    return dict(mean=x.mean(axis=0), scale=np.maximum(x.std(axis=0), .001))


def transform(features, state):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != sum(BLOCKS) or not np.isfinite(x).all():
        raise ValueError('Invalid prediction features')
    z = np.clip((x - state['mean']) / state['scale'], -8., 8.)
    offset = 0
    for width in BLOCKS:
        z[:, offset:offset+width] /= np.sqrt(width)
        offset += width
    return z


def objective(theta, x, y, weights, penalty):
    logits = x @ theta[:-1] + theta[-1]
    loss = np.dot(weights, np.logaddexp(0., logits) - y*logits)
    loss += .5 * penalty * np.dot(theta[:-1], theta[:-1])
    error = weights * (expit(logits) - y)
    gradient = np.r_[x.T @ error + penalty*theta[:-1], error.sum()]
    return float(loss), gradient


def fit(features, labels, eligible, quality, penalty):
    if penalty not in PENALTIES:
        raise ValueError('Undeclared regularization choice')
    state = standardizer(features)
    x = transform(features, state)
    y = (np.asarray(labels) > .5).astype(np.float64)
    weights = stratum_weights(y, eligible, quality)
    result = minimize(objective, np.zeros(x.shape[1]+1), args=(x,y,weights,penalty),
        method='L-BFGS-B', jac=True, options=dict(maxiter=300, ftol=1e-12, gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Convex head did not converge: '+str(result.message))
    return dict(**state, coefficients=result.x[:-1], intercept=float(result.x[-1]),
                penalty=penalty, iterations=int(result.nit), objective=float(result.fun))


def predict(features, state):
    scores = transform(features, state) @ state['coefficients'] + state['intercept']
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite head predictions')
    return scores
