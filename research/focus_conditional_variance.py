"""Fitting-only log-linear heteroscedastic motion uncertainty."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.focus_conditional_motion import predict as predict_mean, FEATURES
from research.focus_residual_calibration import VARIANCE_FLOOR


def design_matrix(mean_model, x):
    x = np.asarray(x, float)
    center, scale = np.asarray(mean_model['center']), np.asarray(mean_model['scale'])
    if (mean_model['features'] != list(FEATURES) or x.ndim != 2 or x.shape[1] != 6
            or center.shape != (6,) or scale.shape != (6,) or (scale <= 0).any()
            or not all(np.isfinite(a).all() for a in (x, center, scale))):
        raise ValueError('Exact finite six-feature normalization required')
    return np.column_stack([np.ones(len(x)), (x - center) / scale])


def objective(theta_flat, design, squared_error):
    theta = np.asarray(theta_flat).reshape(7, 3)
    eta = design @ theta
    logvar = np.log(VARIANCE_FLOOR) + np.logaddexp(0., eta)
    standardized_squared = squared_error * np.exp(-logvar)
    loss = .5 * np.sum(np.log(2 * np.pi) + logvar + standardized_squared) + .5 * np.sum(theta[1:] ** 2)
    gradient = design.T @ (.5 * expit(eta) * (1 - standardized_squared))
    gradient[1:] += theta[1:]
    return loss, gradient.ravel()


def fit(mean_model, x, y, role):
    if role != 'fitting':
        raise ValueError('Only fitting data may enter optimizer')
    x, y = np.asarray(x, float), np.asarray(y, float)
    design = design_matrix(mean_model, x)
    if len(x) < 100 or y.shape != (len(x), 3) or not np.isfinite(y).all():
        raise ValueError('At least100 finite aligned residuals required')
    mean = predict_mean(mean_model, x)
    squared = (y - mean) ** 2
    initial = np.zeros((7, 3))
    initial[0] = np.log(np.maximum(np.asarray(mean_model['variance_um2']) / VARIANCE_FLOOR - 1., 1e-8))
    result = minimize(objective, initial.ravel(), args=(design, squared), jac=True,
                      method='L-BFGS-B', options=dict(maxiter=1000, gtol=1e-8, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Conditional variance optimizer failed: {result.message}')
    return dict(mean_model=mean_model, coefficients=result.x.reshape(7, 3).tolist(),
                variance_floor_um2=VARIANCE_FLOOR.tolist(), ridge=1., intercept_penalized=False,
                role=role, observations=len(x), iterations=int(result.nit), objective=float(result.fun))


def predict(model, x):
    if model['role'] != 'fitting' or model['ridge'] != 1. or model['variance_floor_um2'] != VARIANCE_FLOOR.tolist():
        raise ValueError('Fixed fitted variance contract required')
    theta = np.asarray(model['coefficients'], float)
    if theta.shape != (7, 3) or not np.isfinite(theta).all():
        raise ValueError('Finite aligned uncertainty parameters required')
    eta = design_matrix(model['mean_model'], x) @ theta
    logvar = np.log(VARIANCE_FLOOR) + np.logaddexp(0., eta)
    with np.errstate(over='raise', invalid='raise'):
        variance = np.exp(logvar)
    if not np.isfinite(variance).all() or (variance < VARIANCE_FLOOR * (1 - 1e-12)).any():
        raise ValueError('Finite native-floor constrained uncertainty required')
    return variance


def metrics(y, mean, variance):
    y, mean, variance = (np.asarray(a, float) for a in (y, mean, variance))
    if (y.ndim != 2 or y.shape[1] != 3 or mean.shape != y.shape or variance.shape != y.shape
            or (variance <= 0).any() or not all(np.isfinite(a).all() for a in (y, mean, variance))):
        raise ValueError('Complete finite conditional Gaussian evaluation required')
    error = y - mean
    nll = .5 * np.sum(np.log(2 * np.pi * variance) + error ** 2 / variance, axis=1)
    return dict(observations=len(y), nll_sum=float(nll.sum()), squared_error_sum=float((error ** 2).sum()),
                nll=float(nll.mean()), mse_um2=float((error ** 2).sum(axis=1).mean()))


def gate(rows):
    if len(rows) != 14 or len({r['stem'] for r in rows}) != 14:
        raise ValueError('Fourteen unique held-out movies required')
    if any(r['candidate']['observations'] != r['baseline']['observations'] for r in rows):
        raise ValueError('Identical held-out observation coverage required')
    pooled = {}
    for arm in ('baseline', 'candidate'):
        n = sum(r[arm]['observations'] for r in rows)
        pooled[arm] = dict(observations=n, nll=sum(r[arm]['nll_sum'] for r in rows) / n,
                           mse_um2=sum(r[arm]['squared_error_sum'] for r in rows) / n)
    conditions = dict(pooled_nll_improves=pooled['candidate']['nll'] < pooled['baseline']['nll'],
                      eight_movie_nll_gains=sum(r['candidate']['nll'] < r['baseline']['nll'] for r in rows) >= 8,
                      worst_movie_nll_not_higher=max(r['candidate']['nll'] for r in rows) <= max(r['baseline']['nll'] for r in rows),
                      mean_squared_error_exactly_unchanged=all(r['candidate']['squared_error_sum'] == r['baseline']['squared_error_sum'] for r in rows))
    return dict(pooled=pooled, conditions=conditions, passed=all(conditions.values()))
