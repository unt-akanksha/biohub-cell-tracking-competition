"""The frozen fitting objective/optimizer with a verified resident data provider."""
import time

import numpy as np
from scipy.optimize import minimize

from research.focus_pair_appearance_head import objective, validate_samples
from research.focus_pair_appearance_prepared import Prepared
from research.focus_pair_appearance_resident import Resident


def fit(samples, projection, arm, directory, role, progress=None):
    if role != 'fitting':
        raise ValueError('Only fitting groups may enter the resident head')
    counts = validate_samples(samples)
    if min(counts['parents'], counts['absent']) <= 0:
        raise ValueError('Both original target classes required')
    tick = time.monotonic()
    prepared = Prepared(samples, projection, arm, directory, role)
    provider = Resident(prepared)
    preparation_seconds = time.monotonic() - tick
    weight = float(np.sqrt(counts['parents'] / counts['absent']))
    dimension = prepared.dimension
    bounds = [(None, None)] * dimension
    bounds[4:7] = [(None, .5)] * 3
    initial, _ = objective(np.zeros(dimension), provider, weight)
    iterations = 0
    tick = time.monotonic()

    def callback(theta):
        nonlocal iterations
        iterations += 1
        if progress and iterations % 50 == 0:
            progress(dict(iterations=iterations, fit_elapsed_seconds=time.monotonic()-tick))

    result = minimize(objective, np.zeros(dimension), args=(provider, weight), jac=True,
                      method='L-BFGS-B', bounds=bounds, callback=callback,
                      options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all() or result.fun > initial + 1e-8:
        raise ValueError(f'Frozen resident optimizer did not converge: {result.message}')
    # Identical portable model contract to the original streamed fitter.
    model = dict(arm=arm, role=role, projection=projection, counts=counts, theta=result.x.tolist(),
                 ridge=1., null_logit=-4.5, squared_correction_upper_bound=.5,
                 absent_weight=weight, parent_weight=1., initial_objective=initial,
                 objective=float(result.fun), iterations=int(result.nit), evaluations=int(result.nfev))
    execution = dict(preparation_seconds=preparation_seconds, fit_seconds=time.monotonic()-tick,
                     mode=provider.mode, prepared_disk_bytes=prepared.bytes)
    return model, execution
