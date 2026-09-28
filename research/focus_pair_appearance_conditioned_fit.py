"""Bound-preserving numerical recovery of the same full-feature convex objective."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from research.focus_pair_appearance_head import objective, validate_samples
from research.focus_pair_appearance_prepared import Prepared
from research.focus_pair_appearance_resident import Resident
from research.focus_pair_appearance_conditioning import hessian, coordinate_map


def fit(samples, projection, directory, role, prepared=None, progress=None):
    if role != 'fitting':
        raise ValueError('Only fitting samples may enter numerical recovery')
    counts = validate_samples(samples)
    if (min(counts['parents'], counts['absent']) <= 0
            or projection['real_pair_counts'] != [counts['choices']-counts['groups']-counts['parents'], counts['parents']]):
        raise ValueError('Exact fitting classes and fold-only projection required')
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    if prepared is None:
        prepared = Prepared(samples, projection, 'full', directory / 'prepared', 'fitting')
    provider = Resident(prepared)
    weight = float(np.sqrt(counts['parents']/counts['absent']))
    initial, _ = objective(np.zeros(72), provider, weight)
    curvature, coverage = hessian(np.zeros(72), provider, weight, 'fitting')
    transform, bounds = coordinate_map(curvature, 'fitting')
    before = np.linalg.eigvalsh(curvature)
    after = np.linalg.eigvalsh(transform.T @ curvature @ transform)
    if before[0] <= 0 or after[0] <= 0:
        raise ValueError('Strictly positive fitting curvature required')
    direction = np.arange(1., 73.)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    numerical = (objective(eps*direction, provider, weight)[1]-objective(-eps*direction, provider, weight)[1])/(2*eps)
    predicted = curvature @ direction
    hessian_error = float(np.linalg.norm(predicted-numerical)/max(1., np.linalg.norm(predicted)))
    if hessian_error > 2e-6:
        raise ValueError('Actual full-fold curvature finite-difference check failed')
    np.savez_compressed(directory / 'coordinates.npz', hessian=curvature, transform=transform)
    setup_seconds = time.monotonic()-started
    iteration = 0
    fit_started = time.monotonic()

    def function(beta):
        value, gradient = objective(transform @ beta, provider, weight)
        return value, transform.T @ gradient

    def callback(beta):
        nonlocal iteration
        iteration += 1
        if iteration % 25 == 0:
            record = dict(iteration=iteration, beta=beta.tolist(), theta=(transform @ beta).tolist(),
                          elapsed_seconds=time.monotonic()-fit_started, authorized_for_submission=False)
            (directory / f'iterate-{iteration:06d}.json').write_text(json.dumps(record, allow_nan=False))
            if progress:
                progress(dict(iterations=iteration, fit_elapsed_seconds=record['elapsed_seconds']))

    result = minimize(function, np.zeros(72), jac=True, method='L-BFGS-B', bounds=bounds, callback=callback,
                      options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
    theta = transform @ result.x
    value, gradient = objective(theta, provider, weight)
    state = dict(success=bool(result.success), message=str(result.message), beta=result.x.tolist(), theta=theta.tolist(),
                 objective=float(value), optimizer_objective=float(result.fun), iterations=int(result.nit),
                 evaluations=int(result.nfev), original_gradient=gradient.tolist(),
                 elapsed_seconds=time.monotonic()-fit_started, authorized_for_submission=False)
    (directory / 'optimizer-terminal.json').write_text(json.dumps(state, indent=2, allow_nan=False))
    if (not result.success or not np.isfinite(theta).all() or (theta[4:7] > .5).any()
            or value > initial+1e-8 or abs(value-result.fun) > 1e-7):
        raise ValueError(f'Conditioned original objective did not converge: {result.message}')
    model = dict(arm='full', role='fitting', projection=projection, counts=counts, theta=theta.tolist(),
                 ridge=1., null_logit=-4.5, squared_correction_upper_bound=.5,
                 absent_weight=weight, parent_weight=1., initial_objective=initial, objective=float(value),
                 iterations=int(result.nit), evaluations=int(result.nfev))
    execution = dict(setup_seconds=setup_seconds, fit_seconds=state['elapsed_seconds'], mode=provider.mode,
                     prepared_disk_bytes=prepared.bytes, coverage=coverage, hessian_relative_error=hessian_error,
                     initial_condition=float(before[-1]/before[0]), transformed_condition=float(after[-1]/after[0]))
    return model, execution
