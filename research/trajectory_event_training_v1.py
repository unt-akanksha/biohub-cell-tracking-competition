"""Source-only anchored latent structured learning, with exact training oracles."""
import numpy as np

from research.trajectory_event_assignment_v1 import hinge


def prior(edge_weights, divisions, opportunities, dimensions=30):
    edge_weights = np.asarray(edge_weights, np.float64)
    if (edge_weights.shape != (18,) or not np.isfinite(edge_weights).all()
            or dimensions != 30 or not 0 <= divisions <= opportunities or opportunities <= 0):
        raise ValueError('Invalid source-only event initialization')
    result = np.zeros(dimensions, np.float64)
    result[:18] = edge_weights
    # A regularization anchor from annotated source opportunities, not a claim
    # about the population prevalence or calibrated neural log probabilities.
    result[19] = np.log((divisions + .5) / (opportunities - divisions + .5))
    return result


def fit(cases, anchor, regularization, *, epochs=3, learning_rate=.03,
        seed=20260914, time_limit=5., callback=None):
    """Fixed-budget Adam on partial-label hinge plus an anchored L2 penalty.

    No quality-based stopping/selection. Every provided case is visited each
    epoch. Nonoptimal training oracles fail loudly, never masquerade as labels.
    Callers choose source-only cases and persist each completed epoch.
    """
    anchor = np.asarray(anchor, np.float64)
    penalty = np.asarray(regularization, np.float64)
    if (anchor.ndim != 1 or penalty.shape != anchor.shape or not np.isfinite(anchor).all()
            or not np.isfinite(penalty).all() or np.any(penalty <= 0)
            or not isinstance(epochs, int) or epochs <= 0
            or not np.isfinite(learning_rate) or learning_rate <= 0 or not len(cases)):
        raise ValueError('Invalid anchored optimization contract')
    if any(case['x'].shape[1] != len(anchor) for case in cases):
        raise ValueError('Training feature schemas differ')
    weights, mean, variance = anchor.copy(), np.zeros_like(anchor), np.zeros_like(anchor)
    rng = np.random.default_rng(seed)
    steps, history = 0, []
    for epoch in range(epochs):
        values = []
        for index in rng.permutation(len(cases)):
            value, gradient = hinge(cases[index], weights, time_limit=time_limit)
            delta = weights - anchor
            values.append(float(value + .5 * (penalty * delta) @ delta))
            gradient = gradient + penalty * delta
            gradient *= min(1., 5. / max(float(np.linalg.norm(gradient)), 1e-12))
            steps += 1
            mean = .9 * mean + .1 * gradient
            variance = .999 * variance + .001 * gradient * gradient
            weights -= learning_rate * (mean / (1 - .9**steps)) / (np.sqrt(variance / (1 - .999**steps)) + 1e-8)
            if not np.isfinite(weights).all() or np.linalg.norm(weights) > 100:
                raise ValueError('Event optimizer diverged')
        row = dict(epoch=epoch + 1, steps=steps, cases=len(cases),
                   online_objective_mean=float(np.mean(values)),
                   displacement_norm=float(np.linalg.norm(weights - anchor)))
        history.append(row)
        if callback is not None:
            callback(weights.copy(), dict(row))
    return weights, history
