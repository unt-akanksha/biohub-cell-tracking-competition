"""Full-candidate softmax with training-only square-root absent-class weight."""
import numpy as np
from scipy.optimize import minimize

from research.focus_candidate_ranker import FEATURES, NULL_LOGIT, validate


def class_weights(data, role):
    if role != 'fitting':
        raise ValueError('Only fitting groups may define class weights')
    labels = np.asarray(data['present'])
    if labels.dtype != np.int64 or labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError('Exact binary parent-presence labels required')
    parents = int(labels.sum())
    absent = len(labels) - parents
    if min(parents, absent) <= 0:
        raise ValueError('Both fitting classes required; no fallback weight')
    weight = float(np.sqrt(parents / absent))
    return np.where(labels == 1, 1., weight), weight


def objective(theta, data, group_ids, group_weights):
    x, starts = data['features'], data['starts']
    scores = data['offset'] + x @ theta
    maximum = np.maximum.reduceat(scores, starts)
    weights = np.exp(scores - maximum[group_ids])
    total = np.add.reduceat(weights, starts)
    nll = maximum + np.log(total) - scores[data['chosen']]
    loss = group_weights @ nll + .5 * np.dot(theta[1:], theta[1:])
    gradient = x.T @ (weights / total[group_ids] * group_weights[group_ids])
    gradient -= x[data['chosen']].T @ group_weights
    gradient[1:] += theta[1:]
    return float(loss), gradient


def fit(data, role):
    if role != 'fitting':
        raise ValueError('Only fitting groups may enter optimizer')
    ids = validate(data)
    weights, absent_weight = class_weights(data, role)
    bounds = [(None, None)] * 8
    bounds[4:7] = [(None, .5)] * 3
    initial, _ = objective(np.zeros(8), data, ids, weights)
    result = minimize(objective, np.zeros(8), args=(data, ids, weights), jac=True,
                      method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all() or result.fun > initial + 1e-8:
        raise ValueError(f'Balanced candidate optimizer failed: {result.message}')
    return dict(features=list(FEATURES), theta=result.x.tolist(), ridge=1., role=role,
                null_logit=NULL_LOGIT, observations=len(data['starts']), choices=len(data['offset']),
                fitting_parent=int(data['present'].sum()),
                fitting_absent=int((data['present'] == 0).sum()), parent_weight=1.,
                absent_weight=absent_weight, weight_rule='sqrt(fitting_parent/fitting_absent)',
                initial_objective=initial, objective=float(result.fun), iterations=int(result.nit),
                evaluations=int(result.nfev), squared_correction_upper_bound=.5)


def error_counts(data, model):
    ids = validate(data)
    score = data['offset'] + data['features'] @ np.asarray(model['theta'])
    row_ids = np.arange(len(score))
    maximum = np.maximum.reduceat(score, data['starts'])
    selected = np.minimum.reduceat(np.where(score == maximum[ids], row_ids, len(score)), data['starts'])
    real_score = score.copy()
    real_score[data['null_rows']] = -np.inf
    real_max = np.maximum.reduceat(real_score, data['starts'])
    real_selected = np.minimum.reduceat(
        np.where((real_score == real_max[ids]) & np.isfinite(real_score), row_ids, len(score)), data['starts'])
    parent = data['present'].astype(bool)
    return dict(parent_ranking_ceiling=int((parent & (real_selected == data['chosen'])).sum()),
                parent_rejected=int((parent & (selected == data['null_rows'])).sum()),
                parent_wrong_source=int((parent & (selected != data['null_rows']) & (selected != data['chosen'])).sum()),
                absent_linked=int((~parent & (selected != data['null_rows'])).sum()))


def frame_packet(data, frame):
    groups = np.flatnonzero(data['source_frame'] == frame)
    if not len(groups) or not np.array_equal(groups, np.arange(groups[0], groups[-1] + 1)):
        raise ValueError('A complete contiguous original fitting frame required')
    begin = int(data['starts'][groups[0]])
    end = int(data['starts'][groups[-1]] + data['sizes'][groups[-1]])
    out = {}
    for key, value in data.items():
        out[key] = value[begin:end].copy() if key in ('features', 'offset') else value[groups].copy()
        if key in ('starts', 'chosen', 'null_rows'):
            out[key] -= begin
    validate(out)
    return out
