"""Full residual covariance with a physical PSD floor; no mean change."""
import numpy as np
from research.focus_residual_calibration import VARIANCE_FLOOR


def covariance(errors):
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 3 or len(errors) < 100 or not np.isfinite(errors).all():
        raise ValueError('At least 100 finite three-dimensional residuals required')
    floor = np.sqrt(np.asarray(VARIANCE_FLOOR, dtype=float))
    white = errors / floor
    moment = white.T @ white / len(white)
    values, vectors = np.linalg.eigh(moment)
    result = (vectors * np.maximum(values, 1.)) @ vectors.T
    result = result * floor[:, None] * floor[None, :]
    return (result + result.T) * .5


def metrics(y, mean, cov):
    y, mean, cov = [np.asarray(a, dtype=float) for a in (y, mean, cov)]
    if (y.ndim != 2 or y.shape[1] != 3 or len(y) == 0 or mean.shape not in ((3,), y.shape)
            or cov.shape != (3, 3) or not all(np.isfinite(a).all() for a in (y, mean, cov))
            or not np.allclose(cov, cov.T, rtol=0, atol=1e-12)):
        raise ValueError('Finite aligned Gaussian geometry and symmetric covariance required')
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Positive definite covariance required') from exc
    errors = y - mean
    standardized = np.linalg.solve(factor, errors.T)
    nll = .5 * (3 * np.log(2 * np.pi) + 2 * np.log(np.diag(factor)).sum()
                + (standardized ** 2).sum(axis=0))
    return dict(observations=len(y), nll_sum=float(nll.sum()),
                squared_error_sum=float((errors ** 2).sum()), nll=float(nll.mean()),
                mse_um2=float((errors ** 2).sum(axis=1).mean()))


def gate(rows):
    if len(rows) != 14 or len({r['stem'] for r in rows}) != 14:
        raise ValueError('Fourteen unique movie folds required')
    for row in rows:
        for arm in ('baseline', 'candidate'):
            value = row[arm]
            if value['observations'] <= 0 or not all(np.isfinite(value[k]) for k in ('nll', 'nll_sum', 'squared_error_sum')):
                raise ValueError('Finite nonempty folds required')
        if row['baseline']['observations'] != row['candidate']['observations']:
            raise ValueError('Paired complete observations required')
    pooled = {}
    for arm in ('baseline', 'candidate'):
        n = sum(r[arm]['observations'] for r in rows)
        pooled[arm] = dict(observations=n, nll=sum(r[arm]['nll_sum'] for r in rows)/n,
                           mse_um2=sum(r[arm]['squared_error_sum'] for r in rows)/n)
    wins = sum(r['candidate']['nll'] < r['baseline']['nll'] for r in rows)
    conditions = dict(pooled_nll_improves=pooled['candidate']['nll'] < pooled['baseline']['nll'],
                      majority_movies_nll_improve=wins >= 8,
                      worst_movie_nll_not_higher=max(r['candidate']['nll'] for r in rows) <= max(r['baseline']['nll'] for r in rows),
                      exact_unchanged_mean_errors=all(r['candidate']['squared_error_sum'] == r['baseline']['squared_error_sum'] for r in rows))
    return dict(pooled=pooled, movie_nll_wins=wins, conditions=conditions, passed=all(conditions.values()))
