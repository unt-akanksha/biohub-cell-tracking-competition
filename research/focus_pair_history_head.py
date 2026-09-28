"""A bounded convex candidate ranker with seven previous-image-motion cues."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from research.focus_candidate_ranker import FEATURES as BASE_FEATURES, pack as base_pack, validate as base_validate
from research.focus_balanced_candidate_ranker import class_weights, objective
from research.focus_pair_history import FEATURES as HISTORY_FEATURES, candidate_history
from research.focus_pair_appearance_conditioning import hessian

FEATURES = BASE_FEATURES + HISTORY_FEATURES
CONSTRAINED = np.array([4, 5, 6, 12, 13, 14])
UPPER = np.array([.5, .5, .5, 0., 0., 0.])
MAX_BYTES = 2*1024**3


def pack(packet, previous, parameters, role):
    if role != 'fitting':
        raise ValueError('Only original fitting groups may enter supervised packing')
    data = base_pack(packet, parameters, role)
    columns = np.flatnonzero(np.asarray(packet['labels']) >= 0)
    extra = candidate_history(packet, previous, parameters, columns)['features']
    data['features'] = np.column_stack([data['features'], extra])
    return data


def validate(data):
    x = np.asarray(data['features'])
    if x.ndim != 2 or x.shape[1] != 15 or not np.isfinite(x).all():
        raise ValueError('Finite full15-dimensional candidate features required')
    if sum(a.nbytes for a in data.values()) > MAX_BYTES:
        raise ValueError('Complete fitting arrays exceed2GiB; no candidate truncation')
    ids = base_validate(dict(data, features=x[:, :8]))
    if (np.any(x[data['null_rows'], 8:] != 0) or not np.isin(x[:, 8], [0., 1.]).all()
            or np.any(x[x[:, 8] == 0, 9:] != 0)):
        raise ValueError('Explicit zero null/boundary history and binary availability required')
    if not np.array_equal(x[:, 12:15], x[:, 9:12]**2):
        raise ValueError('Exact declared squared history features required')
    return ids


def blocks(data):
    for first in range(0, len(data['starts']), 64):
        stop = min(first+64, len(data['starts']))
        begin = data['starts'][first]
        end = data['starts'][stop-1]+data['sizes'][stop-1]
        sizes = data['sizes'][first:stop]
        yield dict(x=data['features'][begin:end], offset=data['offset'][begin:end],
            starts=data['starts'][first:stop]-begin, sizes=sizes,
            chosen=data['chosen'][first:stop]-begin, present=data['present'][first:stop],
            ids=np.repeat(np.arange(stop-first), sizes))


def coordinate_map(curvature, role):
    if role != 'fitting':
        raise ValueError('Only fitting curvature may condition the optimizer')
    curvature = np.asarray(curvature, float)
    if (curvature.shape != (15, 15) or not np.isfinite(curvature).all()
            or not np.allclose(curvature, curvature.T, atol=1e-10, rtol=1e-10)):
        raise ValueError('Finite symmetric15D fitting curvature required')
    free = np.array([i for i in range(15) if i not in CONSTRAINED])
    uu = curvature[np.ix_(free, free)]
    uc = curvature[np.ix_(free, CONSTRAINED)]
    whiten = np.linalg.solve(np.linalg.cholesky(uu).T, np.eye(len(free)))
    compensate = -np.linalg.solve(uu, uc)
    schur = curvature[np.ix_(CONSTRAINED, CONSTRAINED)] + curvature[np.ix_(CONSTRAINED, free)] @ compensate
    if np.any(np.diag(schur) <= 0):
        raise ValueError('Positive constrained curvature required; no damping fallback')
    scale = 1/np.sqrt(np.diag(schur))
    transform = np.zeros((15, 15))
    transform[np.ix_(free, free)] = whiten
    transform[np.ix_(free, CONSTRAINED)] = compensate*scale
    transform[CONSTRAINED, CONSTRAINED] = scale
    bounds = [(None, None)]*15
    for i, cap, value in zip(CONSTRAINED, UPPER, scale):
        bounds[i] = (None, float(np.nextafter(cap/value, -np.inf)) if cap else 0.)
    return transform, bounds


def metrics(data, model):
    ids = validate(data)
    theta = np.asarray(model['theta'], float)
    if (model['features'] != list(FEATURES) or model['role'] != 'fitting'
            or model['ridge'] != 1. or model['null_logit'] != -4.5 or theta.shape != (15,)
            or not np.isfinite(theta).all() or np.any(theta[CONSTRAINED] > UPPER)):
        raise ValueError('Exact finite bounded history ranker required')
    score = data['offset']+data['features'] @ theta
    maximum = np.maximum.reduceat(score, data['starts'])
    lse = maximum+np.log(np.add.reduceat(np.exp(score-maximum[ids]), data['starts']))
    rows = np.arange(len(score))
    selected = np.minimum.reduceat(np.where(score == maximum[ids], rows, len(score)), data['starts'])
    correct, present = selected == data['chosen'], data['present'].astype(bool)
    loss = float((lse-score[data['chosen']]).sum())
    return dict(loss_sum=loss, nll=loss/len(present), known_parent=int(present.sum()),
        known_absent=int((~present).sum()), correct_parent=int((correct & present).sum()),
        correct_absent=int((correct & ~present).sum()))


def fit(data, directory, role, progress=None):
    if role != 'fitting':
        raise ValueError('Only original fitting groups may enter optimization')
    ids = validate(data)
    weights, absent_weight = class_weights(data, role)
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    initial, _ = objective(np.zeros(15), data, ids, weights)
    curvature, coverage = hessian(np.zeros(15), lambda: blocks(data), absent_weight, role)
    transform, bounds = coordinate_map(curvature, role)
    eigenvalues = np.linalg.eigvalsh(curvature)
    conditioned = np.linalg.eigvalsh(transform.T @ curvature @ transform)
    if eigenvalues[0] <= 0 or conditioned[0] <= 0:
        raise ValueError('Positive fitting curvature required')
    np.savez_compressed(directory/'coordinates.npz', hessian=curvature, transform=transform)
    iteration = 0

    def function(beta):
        value, gradient = objective(transform @ beta, data, ids, weights)
        return value, transform.T @ gradient

    def callback(beta):
        nonlocal iteration
        iteration += 1
        if iteration % 25 == 0:
            row = dict(iteration=iteration, beta=beta.tolist(), theta=(transform @ beta).tolist())
            (directory/f'iterate-{iteration:06d}.json').write_text(json.dumps(row, allow_nan=False))
            if progress: progress(row)

    optimized = minimize(function, np.zeros(15), jac=True, method='L-BFGS-B', bounds=bounds,
        callback=callback, options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
    theta = transform @ optimized.x
    value, gradient = objective(theta, data, ids, weights)
    terminal = dict(success=bool(optimized.success), message=str(optimized.message), theta=theta.tolist(),
        beta=optimized.x.tolist(), objective=value, optimizer_objective=float(optimized.fun),
        original_gradient=gradient.tolist(), iterations=int(optimized.nit), evaluations=int(optimized.nfev))
    (directory/'optimizer-terminal.json').write_text(json.dumps(terminal, indent=2, allow_nan=False))
    if (not optimized.success or not np.isfinite(theta).all() or np.any(theta[CONSTRAINED] > UPPER)
            or value > initial+1e-8 or abs(value-optimized.fun) > 1e-7):
        raise ValueError(f'Bounded history optimizer failed: {optimized.message}')
    model = dict(features=list(FEATURES), theta=theta.tolist(), role=role, ridge=1., null_logit=-4.5,
        fitting_parent=int(data['present'].sum()), fitting_absent=int((data['present'] == 0).sum()),
        parent_weight=1., absent_weight=absent_weight, weight_rule='sqrt(fitting_parent/fitting_absent)',
        observations=len(data['starts']), choices=len(data['offset']), initial_objective=initial,
        objective=value, iterations=int(optimized.nit), evaluations=int(optimized.nfev),
        squared_correction_upper_bounds=UPPER.tolist(), constrained_indices=CONSTRAINED.tolist(),
        authorized_for_submission=False)
    (directory/'model.json').write_text(json.dumps(model, indent=2, allow_nan=False))
    return model, dict(coverage=coverage, initial_condition=float(eigenvalues[-1]/eigenvalues[0]),
        transformed_condition=float(conditioned[-1]/conditioned[0]))
