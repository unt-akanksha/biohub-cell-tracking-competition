"""Source-trained ordinary motion likelihood and endpoint-only inference."""
import copy
from collections import defaultdict
import numpy as np
from scipy.stats import chi2

SCALE = np.asarray((1.625, .40625, .40625))


def regression_arrays(windows):
    v = np.diff(np.asarray(windows, dtype=float), axis=1)
    forward = np.concatenate((v[:, 0], v[:, 1]), axis=1)
    backward = np.concatenate((-v[:, 4], -v[:, 3]), axis=1)
    return forward, backward, v[:, 2]


def residuals(model, windows):
    forward, backward, middle = regression_arrays(windows)
    def predict(x):
        return np.column_stack((x / model['scale'], np.ones(len(x)))) @ model['coef']
    return np.concatenate((middle - predict(forward), -middle - predict(backward)), axis=1)


def scores(model, windows):
    r = residuals(model, windows) - model['residual_mean']
    return np.einsum('ni,ij,nj->n', r, model['precision'], r)


def fit(windows, movie_ids):
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3 or windows.shape[1:] != (6, 3) or not np.isfinite(windows).all():
        raise ValueError('Finite physical six-point trajectories required')
    forward, backward, middle = regression_arrays(windows)
    _, inverse, counts = np.unique(movie_ids, return_inverse=True, return_counts=True)
    per_window = 1. / counts[inverse]; per_window /= per_window.sum()
    x = np.concatenate((forward, backward)); y = np.concatenate((middle, -middle))
    weight = np.tile(per_window / 2, 2)
    scale = np.maximum(np.sqrt((weight[:, None] * x * x).sum(axis=0)), 1e-3)
    x = np.column_stack((x / scale, np.ones(len(x))))
    regularizer = np.diag([.01] * 6 + [1e-10])
    robust = np.ones(len(x))
    floor = np.diag(SCALE ** 2 / 6)
    for _ in range(10):
        w = weight * robust; w /= w.sum()
        coef = np.linalg.solve(x.T @ (w[:, None] * x) + regularizer,
                               x.T @ (w[:, None] * y))
        r = y - x @ coef
        covariance = r.T @ (w[:, None] * r) + floor
        distance = np.einsum('ni,ij,nj->n', r, np.linalg.inv(covariance), r)
        robust = np.minimum(1., np.sqrt(chi2.ppf(.95, 3) / np.maximum(distance, 1e-12)))
    model = dict(scale=scale, coef=coef)
    joint = residuals(model, windows)
    joint_weight = per_window * np.sqrt(robust[:len(windows)] * robust[len(windows):])
    joint_weight /= joint_weight.sum()
    mean = (joint_weight[:, None] * joint).sum(axis=0)
    centered = joint - mean
    covariance = centered.T @ (joint_weight[:, None] * centered) + np.diag(np.tile(SCALE ** 2 / 6, 2))
    model.update(residual_mean=mean, precision=np.linalg.inv(covariance), covariance=covariance)
    return model


def calibrate(model, windows):
    values = np.sort(scores(model, windows))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite source calibration scores required')
    index = min(len(values), int(np.ceil((len(values) + 1) * .99))) - 1
    return float(values[index])


def reconnect(final, coords, raw_edges, detector_ids, model, threshold):
    nodes = {int(i): n for i, n in final['nodes'].items()}
    genuine = set(map(int, detector_ids)) & set(nodes)
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for e in final['edges']:
        a, b = e['source_id'], e['target_id']
        if nodes[b]['t'] != nodes[a]['t'] + 1:
            raise ValueError('Nonadjacent final edge')
        outgoing[a].append(b); incoming[b].append(a)
    if max(map(len, incoming.values()), default=0) > 1 or max(map(len, outgoing.values()), default=0) > 2:
        raise ValueError('Invalid lineage degrees')
    forward, backward = defaultdict(dict), defaultdict(dict)
    for a, b, p, _ in np.asarray(raw_edges):
        if not np.isfinite([a, b, p]).all() or a != int(a) or b != int(b):
            raise ValueError('Invalid raw edge')
        a, b = int(a), int(b)
        if not (0 <= a < len(coords) and 0 <= b < len(coords) and 0 <= p <= 1):
            raise ValueError('Invalid raw edge range')
        if coords[b, 0] != coords[a, 0] + 1:
            raise ValueError('Nonadjacent raw edge')
        forward[a][b] = max(forward[a].get(b, 0.), p)
        backward[b][a] = max(backward[b].get(a, 0.), p)
    def best(table):
        result = {}
        for a, choices in table.items():
            p = max(choices.values()); winners = [b for b, v in choices.items() if v == p]
            if p >= .88 and len(winners) == 1:
                result[a] = (winners[0], p)
        return result
    fb, bb = best(forward), best(backward)
    def chain(start, reverse):
        result = [start]
        lookup, reciprocal = (incoming, outgoing) if reverse else (outgoing, incoming)
        for _ in range(2):
            choices = lookup[result[-1]]
            if len(choices) != 1 or len(reciprocal[choices[0]]) != 1:
                return None
            result.append(choices[0])
        return result
    records = []; eligible = 0
    for a, (b, probability) in sorted(fb.items()):
        if bb.get(b, (None,))[0] != a or a not in genuine or b not in genuine or outgoing[a] or incoming[b]:
            continue
        past, future = chain(a, True), chain(b, False)
        if past is None or future is None:
            continue
        eligible += 1
        path = past[::-1] + future
        window = np.asarray([[nodes[i][k] for k in ('z', 'y', 'x')] for i in path]) * SCALE
        score = float(scores(model, window[None])[0])
        if score <= threshold:
            records.append(dict(source_id=a, target_id=b, probability=float(probability),
                                motion_score=score, source_threshold=threshold))
    if len({r['source_id'] for r in records}) != len(records) or len({r['target_id'] for r in records}) != len(records):
        raise ValueError('Overlapping reciprocal endpoints')
    output = copy.deepcopy(final)
    output['edges'].extend(dict(source_id=r['source_id'], target_id=r['target_id']) for r in records)
    return output, dict(eligible_endpoint_pairs=eligible, added_edges=len(records), added_nodes=0,
                        links=records, existing_output_unchanged=True, ground_truth_used=False)
