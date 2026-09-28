"""Convex candidate-vs-null ranking with full candidate coverage."""
import numpy as np
from scipy.optimize import minimize

SCALE = np.array([1.625, .40625, .40625])
NULL_LOGIT = -4.5
FEATURES = ('real_intercept', 'delta_z', 'delta_y', 'delta_x',
            'delta_z_squared', 'delta_y_squared', 'delta_x_squared', 'feature_cosine')
MAX_BYTES = 2 * 1024 ** 3


def candidate_arrays(packet, parameters, columns=None):
    source, target = np.asarray(packet['source_coords'], float), np.asarray(packet['target_coords'], float)
    sf, tf = np.asarray(packet['source_features'], float), np.asarray(packet['target_features'], float)
    flow = np.asarray(packet['backward_um'], float)
    ns, nt = len(source), len(target)
    mean, variance = np.asarray(parameters['mean_um'], float), np.asarray(parameters['variance_um2'], float)
    if (source.shape != (ns, 3) or target.shape != (nt, 3) or sf.shape != (ns, 32)
            or tf.shape != (nt, 32) or flow.shape != (nt, 3) or mean.shape != (3,)
            or variance.shape != (3,) or (variance <= 0).any() or max(ns, nt) > 2048
            or not all(np.isfinite(a).all() for a in (source, target, sf, tf, flow, mean, variance))):
        raise ValueError('Finite complete bounded candidate geometry/features required')
    if columns is None:
        columns = np.arange(nt, dtype=np.int64)
    columns = np.asarray(columns)
    if (columns.dtype != np.int64 or columns.ndim != 1 or len(set(columns)) != len(columns)
            or (columns < 0).any() or (columns >= nt).any()):
        raise ValueError('Unique original target columns required')
    ng = len(columns)
    if ng * (ns + 1) * 9 * 8 > MAX_BYTES:
        raise ValueError('Candidate allocation exceeds2GiB; no truncation fallback')
    delta = ((source[None] - target[columns, None]) * SCALE - flow[columns, None] - mean) / np.sqrt(variance)
    sn = sf / np.maximum(np.linalg.norm(sf, axis=1, keepdims=True), 1e-12)
    tn = tf[columns] / np.maximum(np.linalg.norm(tf[columns], axis=1, keepdims=True), 1e-12)
    cosine = tn @ sn.T
    x = np.zeros((ng, ns + 1, len(FEATURES)))
    x[:, :ns, 0] = 1.
    x[:, :ns, 1:4] = delta
    x[:, :ns, 4:7] = delta ** 2
    x[:, :ns, 7] = cosine
    offset = np.full((ng, ns + 1), NULL_LOGIT)
    offset[:, :ns] = -.5 * np.sum(delta ** 2, axis=-1)
    starts = np.arange(ng, dtype=np.int64) * (ns + 1)
    return dict(features=x.reshape(-1, len(FEATURES)), offset=offset.ravel(), starts=starts,
                sizes=np.full(ng, ns + 1, dtype=np.int64), null_rows=starts + ns)


def pack(packet, parameters, role):
    if role != 'fitting':
        raise ValueError('Only original fitting targets may enter supervised packing')
    labels = np.asarray(packet['labels'])
    ns, nt = len(packet['source_coords']), len(packet['target_coords'])
    if labels.shape != (nt,) or labels.dtype != np.int64 or (labels < -1).any() or (labels > ns).any():
        raise ValueError('Exact parent/null/unknown labels required')
    columns = np.flatnonzero(labels >= 0)
    result = candidate_arrays(packet, parameters, columns)
    result.update(chosen=result['starts'] + labels[columns], present=(labels[columns] < ns).astype(np.int64),
                  target_indices=np.asarray(packet['target_indices'])[columns].copy(),
                  source_frame=np.full(len(columns), int(packet['source_frame']), np.int64))
    return result


def combine(parts):
    parts = [part for part in parts if len(part['starts'])]
    if not parts:
        raise ValueError('Nonempty supervised choice groups required')
    if sum(sum(a.nbytes for a in part.values()) for part in parts) > MAX_BYTES:
        raise ValueError('Packed array allocation exceeds2GiB')
    offsets = np.cumsum([0] + [len(p['offset']) for p in parts[:-1]])
    shifted = {'starts', 'chosen', 'null_rows'}
    return {key: np.concatenate([part[key] + shift if key in shifted else part[key]
                                for part, shift in zip(parts, offsets)]) for key in parts[0]}


def validate(data):
    x, offset, starts, sizes = (np.asarray(data[k]) for k in ('features', 'offset', 'starts', 'sizes'))
    n, ng = len(offset), len(starts)
    expected = np.cumsum(np.r_[0, sizes[:-1]])
    if (ng == 0 or x.shape != (n, 8) or offset.shape != (n,) or starts.dtype != np.int64
            or sizes.dtype != np.int64 or sizes.shape != (ng,) or (sizes < 1).any()
            or sizes.sum() != n or not np.array_equal(starts, expected)
            or not np.isfinite(x).all() or not np.isfinite(offset).all()):
        raise ValueError('Complete finite contiguous choice groups required')
    for key in ('chosen', 'null_rows', 'present', 'target_indices', 'source_frame'):
        if data[key].shape != (ng,) or data[key].dtype != np.int64:
            raise ValueError('Exact aligned integer group identities required')
    if (not np.array_equal(data['null_rows'], starts + sizes - 1)
            or (data['chosen'] < starts).any() or (data['chosen'] >= starts + sizes).any()
            or not np.array_equal(data['present'], (data['chosen'] != data['null_rows']).astype(np.int64))
            or np.any(x[data['null_rows']] != 0) or np.any(offset[data['null_rows']] != NULL_LOGIT)):
        raise ValueError('Fixed null class and complete known labels required')
    if sum(a.nbytes for a in data.values()) > MAX_BYTES:
        raise ValueError('Packed array budget exceeded')
    return np.repeat(np.arange(ng), sizes)


def objective(theta, data, group_ids):
    x, starts = data['features'], data['starts']
    scores = data['offset'] + x @ theta
    maximum = np.maximum.reduceat(scores, starts)
    weights = np.exp(scores - maximum[group_ids])
    total = np.add.reduceat(weights, starts)
    loss = np.sum(maximum + np.log(total) - scores[data['chosen']]) + .5 * np.dot(theta[1:], theta[1:])
    gradient = x.T @ (weights / total[group_ids]) - x[data['chosen']].sum(axis=0)
    gradient[1:] += theta[1:]
    return float(loss), gradient


def fit(data, role):
    if role != 'fitting':
        raise ValueError('Only fitting groups may enter optimizer')
    group_ids = validate(data)
    bounds = [(None, None)] * 8
    bounds[4:7] = [(None, .5)] * 3
    initial, _ = objective(np.zeros(8), data, group_ids)
    result = minimize(objective, np.zeros(8), args=(data, group_ids), jac=True, method='L-BFGS-B',
                      bounds=bounds, options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all() or result.fun > initial + 1e-8:
        raise ValueError(f'Candidate optimizer failed: {result.message}')
    return dict(features=list(FEATURES), theta=result.x.tolist(), ridge=1., role=role,
                null_logit=NULL_LOGIT, observations=len(data['starts']), choices=len(data['offset']),
                fitting_parent=int(data['present'].sum()), initial_objective=float(initial),
                objective=float(result.fun), iterations=int(result.nit), evaluations=int(result.nfev),
                squared_correction_upper_bound=.5)


def metrics(data, model=None):
    group_ids = validate(data)
    theta = np.zeros(8)
    if model is not None:
        theta = np.asarray(model['theta'], float)
        if (model['features'] != list(FEATURES) or model['role'] != 'fitting' or model['ridge'] != 1.
                or model['null_logit'] != NULL_LOGIT or theta.shape != (8,)
                or not np.isfinite(theta).all() or (theta[4:7] > .5).any()):
            raise ValueError('Fixed finite fitted candidate ranking required')
    score = data['offset'] + data['features'] @ theta
    maxima = np.maximum.reduceat(score, data['starts'])
    lse = maxima + np.log(np.add.reduceat(np.exp(score - maxima[group_ids]), data['starts']))
    row_numbers = np.arange(len(score))
    selected = np.minimum.reduceat(np.where(score == maxima[group_ids], row_numbers, len(score)), data['starts'])
    present = data['present'].astype(bool)
    correct = selected == data['chosen']
    loss = float((lse - score[data['chosen']]).sum())
    return dict(loss_sum=loss, known_parent=int(present.sum()), known_absent=int((~present).sum()),
                correct_parent=int((correct & present).sum()), correct_absent=int((correct & ~present).sum()),
                nll=loss / len(present))
