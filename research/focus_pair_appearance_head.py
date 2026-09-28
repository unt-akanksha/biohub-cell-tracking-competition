"""Bounded-block full candidate softmax; same gates and loss, richer appearance."""
import numpy as np
from scipy.optimize import minimize

from research.focus_candidate_ranker import validate, MAX_BYTES
from research.focus_pair_appearance import transform

GROUP_BLOCK = 64


def blocks(samples, projection, arm, group_block=GROUP_BLOCK):
    if arm not in ('base', 'full', 'lda') or not isinstance(group_block, int) or group_block <= 0:
        raise ValueError('Declared arm and positive complete-group block required')
    for base, appearance in samples:
        ng = len(base['starts'])
        for first in range(0, ng, group_block):
            last = min(first + group_block, ng)
            begin = int(base['starts'][first])
            end = int(base['starts'][last - 1] + base['sizes'][last - 1])
            null = base['null_rows'][first:last] - begin
            x = base['features'][begin:end]
            if arm != 'base':
                extra = transform(appearance[begin:end], null, projection, arm)
                x = np.column_stack([x, extra])
            if x.nbytes > MAX_BYTES:
                raise ValueError('Active complete feature block exceeds2GiB')
            sizes = base['sizes'][first:last]
            yield dict(x=x, offset=base['offset'][begin:end], starts=base['starts'][first:last] - begin,
                       sizes=sizes, null_rows=null, chosen=base['chosen'][first:last] - begin,
                       present=base['present'][first:last], ids=np.repeat(np.arange(last - first), sizes))


def objective(theta, provider, absent_weight):
    loss = 0.
    gradient = np.zeros(len(theta))
    for block in provider():
        x, starts, ids = (block[k] for k in ('x', 'starts', 'ids'))
        scores = block['offset'] + x @ theta
        maxima = np.maximum.reduceat(scores, starts)
        numerator = np.exp(scores - maxima[ids])
        total = np.add.reduceat(numerator, starts)
        weights = np.where(block['present'] == 1, 1., absent_weight)
        loss += weights @ (maxima + np.log(total) - scores[block['chosen']])
        gradient += x.T @ (numerator / total[ids] * weights[ids]) - x[block['chosen']].T @ weights
    loss += .5 * np.dot(theta[1:], theta[1:])
    gradient[1:] += theta[1:]
    return float(loss), gradient


def validate_samples(samples):
    if not samples:
        raise ValueError('Complete nonempty fitting samples required')
    counts = dict(groups=0, parents=0, choices=0)
    for base, appearance in samples:
        validate(base)
        if appearance.shape != (len(base['offset']), 64) or appearance.dtype != np.float32:
            raise ValueError('Exact full float32 descriptor coverage required')
        counts['groups'] += len(base['starts'])
        counts['parents'] += int(base['present'].sum())
        counts['choices'] += len(base['offset'])
    counts['absent'] = counts['groups'] - counts['parents']
    return counts


def fit(samples, projection, arm, role):
    if role != 'fitting':
        raise ValueError('Only fitting samples may enter the appearance head')
    if arm not in ('full', 'lda'):
        raise ValueError('Only the two predeclared new arms may be fitted')
    counts = validate_samples(samples)
    expected = [counts['choices'] - counts['groups'] - counts['parents'], counts['parents']]
    if min(counts['parents'], counts['absent']) <= 0 or projection['real_pair_counts'] != expected:
        raise ValueError('Projection must use the exact same fitting pairs and both target classes')
    absent_weight = float(np.sqrt(counts['parents'] / counts['absent']))
    dim = 72 if arm == 'full' else 9
    provider = lambda: blocks(samples, projection, arm)
    bounds = [(None, None)] * dim
    bounds[4:7] = [(None, .5)] * 3
    initial, _ = objective(np.zeros(dim), provider, absent_weight)
    result = minimize(objective, np.zeros(dim), args=(provider, absent_weight), jac=True,
                      method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all() or result.fun > initial + 1e-8:
        raise ValueError(f'Appearance head did not converge: {result.message}')
    return dict(arm=arm, role=role, projection=projection, counts=counts, theta=result.x.tolist(),
                ridge=1., null_logit=-4.5, squared_correction_upper_bound=.5,
                absent_weight=absent_weight, parent_weight=1., initial_objective=initial,
                objective=float(result.fun), iterations=int(result.nit), evaluations=int(result.nfev))


def metrics(samples, model):
    if (model['role'] != 'fitting' or model['arm'] not in ('full', 'lda') or model['ridge'] != 1.
            or model['null_logit'] != -4.5 or model['squared_correction_upper_bound'] != .5):
        raise ValueError('Exact fitted appearance model required')
    validate_samples(samples)
    theta = np.asarray(model['theta'])
    if (theta.shape != (72 if model['arm'] == 'full' else 9,) or not np.isfinite(theta).all()
            or (theta[4:7] > .5).any()):
        raise ValueError('Finite coefficients and fixed physical bounds required')
    result = dict(loss_sum=0., known_parent=0, known_absent=0, correct_parent=0, correct_absent=0)
    for block in blocks(samples, model['projection'], model['arm']):
        score = block['offset'] + block['x'] @ theta
        maximum = np.maximum.reduceat(score, block['starts'])
        lse = maximum + np.log(np.add.reduceat(np.exp(score - maximum[block['ids']]), block['starts']))
        rows = np.arange(len(score))
        selected = np.minimum.reduceat(np.where(score == maximum[block['ids']], rows, len(rows)), block['starts'])
        parent = block['present'].astype(bool)
        correct = selected == block['chosen']
        result['loss_sum'] += float((lse - score[block['chosen']]).sum())
        result['known_parent'] += int(parent.sum())
        result['known_absent'] += int((~parent).sum())
        result['correct_parent'] += int((correct & parent).sum())
        result['correct_absent'] += int((correct & ~parent).sum())
    result['nll'] = result['loss_sum'] / (result['known_parent'] + result['known_absent'])
    return result
