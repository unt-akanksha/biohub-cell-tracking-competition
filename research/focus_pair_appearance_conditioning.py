"""Exact fitting Hessian and bound-preserving optimizer coordinate transform."""
import numpy as np


def hessian(theta, provider, absent_weight, role):
    if role != 'fitting':
        raise ValueError('Only fitting groups may determine optimizer coordinates')
    theta = np.asarray(theta, float)
    if theta.ndim != 1 or len(theta) < 8 or not np.isfinite(theta).all() or not np.isfinite(absent_weight) or absent_weight <= 0:
        raise ValueError('Finite original parameters and positive fixed weight required')
    result = np.diag(np.r_[0., np.ones(len(theta)-1)])
    choices = active_choices = groups = 0
    for block in provider():
        x, starts, ids = (block[k] for k in ('x', 'starts', 'ids'))
        score = block['offset'] + x @ theta
        maximum = np.maximum.reduceat(score, starts)
        probability = np.exp(score-maximum[ids])
        probability /= np.add.reduceat(probability, starts)[ids]
        weight = np.where(block['present'], 1., absent_weight)
        means = np.add.reduceat(probability[:, None] * x, starts, axis=0)
        active = probability != 0  # Omit exactly zero floating-point terms, never candidates.
        selected = x[active]
        result += selected.T @ ((probability[active] * weight[ids[active]])[:, None] * selected)
        result -= means.T @ (weight[:, None] * means)
        choices += len(x)
        active_choices += int(active.sum())
        groups += len(starts)
    result = (result+result.T)/2
    if not groups or not np.isfinite(result).all():
        raise ValueError('Complete finite fitting curvature required')
    return result, dict(groups=groups, choices=choices, nonzero_probability_choices=active_choices)


def coordinate_map(curvature, role):
    if role != 'fitting':
        raise ValueError('Only fitting curvature may define coordinates')
    matrix = np.asarray(curvature, float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 8
            or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10)):
        raise ValueError('Finite symmetric fitting Hessian required')
    dim = len(matrix)
    constrained = np.array([4, 5, 6])
    free = np.array([i for i in range(dim) if i not in constrained])
    uu = matrix[np.ix_(free, free)]
    uc = matrix[np.ix_(free, constrained)]
    factor = np.linalg.cholesky(uu)
    whiten = np.linalg.solve(factor.T, np.eye(len(free)))
    compensate = -np.linalg.solve(uu, uc)
    schur = matrix[np.ix_(constrained, constrained)] + matrix[np.ix_(constrained, free)] @ compensate
    if (np.diag(schur) <= 0).any():
        raise ValueError('Positive constrained Schur curvature required; no damping fallback')
    scale = 1/np.sqrt(np.diag(schur))
    transform = np.zeros_like(matrix)
    transform[np.ix_(free, free)] = whiten
    transform[np.ix_(free, constrained)] = compensate * scale
    transform[constrained, constrained] = scale
    bounds = [(None, None)] * dim
    for index, value in zip(constrained, scale):
        bounds[index] = (None, float(np.nextafter(.5/value, -np.inf)))
    return transform, bounds


def transformed_objective(beta, transform, objective, provider, absent_weight):
    beta = np.asarray(beta, float)
    theta = transform @ beta
    value, gradient = objective(theta, provider, absent_weight)
    return value, transform.T @ gradient
