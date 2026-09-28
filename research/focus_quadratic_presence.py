"""Fixed polynomial offset-logistic correction; no decision-threshold fitting."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.focus_parent_presence import FEATURES, metrics

PAIRS = tuple((i, j) for i in range(len(FEATURES)) for j in range(i, len(FEATURES)))


def expand(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError('Finite seven-feature context required')
    return np.column_stack([x] + [x[:, i] * x[:, j] for i, j in PAIRS])


def safe_std(x):
    scale = x.std(axis=0)
    return np.where(scale > 1e-8, scale, 1.)


def fit(data, role):
    if role != 'fitting':
        raise ValueError('Only fitting data may enter optimizer')
    x = np.asarray(data['context'], dtype=float)
    y = np.asarray(data['present'], dtype=float)
    offset = np.asarray(data['offset'], dtype=float)
    if (x.shape != (len(y), len(FEATURES)) or y.ndim != 1 or offset.shape != y.shape
            or len(y) < 100 or set(y) != {0., 1.}
            or not all(np.isfinite(a).all() for a in (x, y, offset))):
        raise ValueError('Finite fitting context with both classes required')
    mean, scale = x.mean(axis=0), safe_std(x)
    basis = expand((x - mean) / scale)
    basis_mean, basis_scale = basis.mean(axis=0), safe_std(basis)
    design = np.column_stack([np.ones(len(y)), (basis - basis_mean) / basis_scale])

    def objective(theta):
        eta = offset + design @ theta
        value = np.sum(np.logaddexp(0., eta) - y * eta) + .5 * np.dot(theta[1:], theta[1:])
        grad = design.T @ (expit(eta) - y)
        grad[1:] += theta[1:]
        return value, grad

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=1000, gtol=1e-8, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Quadratic optimizer failed: {result.message}')
    return dict(features=list(FEATURES), pairs=[list(p) for p in PAIRS],
                mean=mean.tolist(), scale=scale.tolist(), basis_mean=basis_mean.tolist(),
                basis_scale=basis_scale.tolist(), theta=result.x.tolist(), role=role,
                fitting_examples=len(y), fitting_present=int(y.sum()), ridge=1.,
                iterations=int(result.nit), objective=float(result.fun),
                intercept_penalized=False)


def predict_offset(data, model):
    if model['features'] != list(FEATURES) or model['pairs'] != [list(p) for p in PAIRS] or model['role'] != 'fitting':
        raise ValueError('Exact fitted feature contract required')
    x = np.asarray(data['context'], dtype=float)
    mean, scale = np.asarray(model['mean']), np.asarray(model['scale'])
    bm, bs = np.asarray(model['basis_mean']), np.asarray(model['basis_scale'])
    theta = np.asarray(model['theta'])
    width = len(FEATURES) + len(PAIRS)
    if (mean.shape != (7,) or scale.shape != (7,) or bm.shape != (width,)
            or bs.shape != (width,) or theta.shape != (width + 1,)
            or not all(np.isfinite(a).all() for a in (mean, scale, bm, bs, theta))
            or (scale <= 0).any() or (bs <= 0).any()):
        raise ValueError('Finite aligned fitted normalization required')
    result = np.asarray(data['offset'], dtype=float) + theta[0] + ((expand((x - mean) / scale) - bm) / bs) @ theta[1:]
    if result.shape != (len(x),) or not np.isfinite(result).all():
        raise ValueError('Finite aligned posterior offsets required')
    return result


def evaluate(data, model):
    return metrics(dict(data, offset=predict_offset(data, model)))


def pooled(rows):
    keys = ('loss_sum', 'known_parent', 'known_absent', 'correct_parent', 'correct_absent')
    result = {key: sum(row[key] for row in rows) for key in keys}
    result['nll'] = result['loss_sum'] / (result['known_parent'] + result['known_absent'])
    return result


def screen(folds):
    if len(folds) != 12 or len({f['held_out'] for f in folds}) != 12:
        raise ValueError('Twelve unique movie-held-out folds required')
    totals = {arm: pooled([f[arm] for f in folds]) for arm in ('original', 'linear', 'quadratic')}
    q, controls = totals['quadratic'], [totals['original'], totals['linear']]
    gates = dict(nll_below_both=q['nll'] < min(c['nll'] for c in controls),
                 parent_not_lower=q['correct_parent'] >= max(c['correct_parent'] for c in controls),
                 absent_not_lower=q['correct_absent'] >= max(c['correct_absent'] for c in controls),
                 eight_movie_nll_wins=sum(f['quadratic']['nll'] < f['linear']['nll'] for f in folds) >= 8,
                 worst_nll_regression=max(f['quadratic']['nll'] - f['linear']['nll'] for f in folds) <= .02)
    return dict(passed=all(gates.values()), gates=gates, pooled=totals,
                scope='Fitting-domain correction LOMO only; not independent tracking validation')
