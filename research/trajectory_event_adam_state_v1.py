"""Serializable Adam state matching the fixed event learner's exact update."""
import numpy as np


def initialize(anchor):
    weights = np.asarray(anchor, np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError('Finite one-dimensional initializer required')
    return dict(weights=weights.copy(), mean=np.zeros_like(weights), variance=np.zeros_like(weights), steps=0)


def update(state, gradient, learning_rate=.03):
    weights = state['weights']
    gradient = np.array(gradient, dtype=np.float64, copy=True)
    if gradient.shape != weights.shape or not np.isfinite(gradient).all():
        raise ValueError('Invalid gradient')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid learning rate')
    if (state['mean'].shape != weights.shape or state['variance'].shape != weights.shape
            or not all(np.isfinite(state[k]).all() for k in ('weights', 'mean', 'variance'))
            or np.any(state['variance'] < 0) or not isinstance(state['steps'], int) or state['steps'] < 0):
        raise ValueError('Invalid optimizer state')
    gradient *= min(1., 5. / max(float(np.linalg.norm(gradient)), 1e-12))
    step = state['steps'] + 1
    mean = .9 * state['mean'] + .1 * gradient
    variance = .999 * state['variance'] + .001 * gradient * gradient
    result = weights - learning_rate * (mean / (1 - .9**step)) / (np.sqrt(variance / (1 - .999**step)) + 1e-8)
    if not np.isfinite(result).all() or np.linalg.norm(result) > 100:
        raise ValueError('Event optimizer diverged')
    state.update(weights=result, mean=mean, variance=variance, steps=step)
    return state
