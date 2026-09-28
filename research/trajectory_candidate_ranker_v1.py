"""Small regularized trajectory ranker; no graph edits or GPU dependency."""
import numpy as np
from scipy.optimize import minimize
from research.trajectory_disagreement_data_v1 import positions, adjacency

FEATURES = ('final_distance', 'raw_distance', 'half_history_residual',
            'full_history_residual', 'half_future_residual', 'full_future_residual',
            'history_speed', 'future_speed', 'velocity_disagreement',
            'history_known', 'future_known', 'both_known',
            'raw_probability', 'raw_log_odds', 'raw_probability_known',
            'initial_edge', 'parent_localization_shift', 'child_localization_shift')


def features(initial, final, groups, raw_edges):
    """Every feature uses predictions only; no current-edge or identity feature."""
    old = {int(k): v for k,v in initial['nodes'].items()}
    nodes = {int(k): v for k,v in final['nodes'].items()}
    pos, raw_pos = positions(nodes), positions(old)
    incoming, outgoing = adjacency(final['edges'])
    cap = max(nodes) + 1
    p = np.zeros((cap, 3)); rp = np.zeros_like(p)
    history = np.zeros_like(p); future = np.zeros_like(p)
    hk = np.zeros(cap, bool); fk = np.zeros(cap, bool)
    for i in nodes:
        p[i] = pos[i]
        if i in raw_pos:
            rp[i] = raw_pos[i]
        if len(incoming[i]) == 1:
            prev = incoming[i][0]
            if int(nodes[i]['t']) - int(nodes[prev]['t']) == 1:
                history[i] = pos[i] - pos[prev]; hk[i] = True
        if len(outgoing[i]) == 1:
            nxt = outgoing[i][0]
            if int(nodes[nxt]['t']) - int(nodes[i]['t']) == 1:
                future[i] = pos[nxt] - pos[i]; fk[i] = True
    parent = groups['parents']; child = np.repeat(groups['children'], np.diff(groups['offsets']))
    if not set(parent) <= old.keys() or not set(child) <= old.keys():
        raise ValueError('Only observed original endpoints supported')
    delta = p[child] - p[parent]
    norm = lambda a: np.linalg.norm(a, axis=1) / 10.
    hmask, fmask = hk[parent].astype(float), fk[child].astype(float)
    raw_edges = np.asarray(raw_edges)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 4 or not np.isfinite(raw_edges).all():
        raise ValueError('Invalid raw edge schema')
    probabilities = {(int(a),int(b)):float(prob) for a,b,prob,_ in raw_edges}
    if len(probabilities) != len(raw_edges) or any(not 0 <= v <= 1 for v in probabilities.values()):
        raise ValueError('Invalid raw probabilities')
    pairs = list(zip(parent.tolist(), child.tolist()))
    prob = np.array([probabilities.get(pair, .5) for pair in pairs])
    known = np.array([pair in probabilities for pair in pairs], float)
    ilp = {(int(e['source_id']),int(e['target_id'])) for e in initial['edges']}
    initial_edge = np.array([pair in ilp for pair in pairs], float)
    clipped = np.clip(prob, .001, .999)
    matrix = np.column_stack((norm(delta), norm(rp[child]-rp[parent]),
        norm(delta-.5*history[parent])*hmask, norm(delta-history[parent])*hmask,
        norm(delta-.5*future[child])*fmask, norm(delta-future[child])*fmask,
        norm(history[parent]), norm(future[child]), norm(history[parent]-future[child])*hmask*fmask,
        hmask, fmask, hmask*fmask, prob, np.log(clipped/(1-clipped))/5., known,
        initial_edge, norm(p[parent]-rp[parent]), norm(p[child]-rp[child])))
    if matrix.shape != (len(parent),len(FEATURES)) or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite features or schema drift')
    return matrix.astype(np.float32)


def supervised(matrix, groups, target, safe):
    result = []
    for i,p in enumerate(target):
        if p < 0:
            continue
        a,b = groups['offsets'][i:i+2]
        selected = np.flatnonzero(safe[a:b])
        if len(selected) < 2:
            continue
        choices = groups['parents'][a:b][selected]
        where = np.flatnonzero(choices == p)
        if len(where) != 1:
            raise ValueError('Missing unique positive')
        result.append(dict(x=matrix[a:b][selected].astype(np.float64), y=int(where[0]),
                           parents=choices, child=int(groups['children'][i]),
                           current=int(groups['current'][i]), initial=int(groups['neural'][i])))
    return result


def fit(rows, regularization=.01):
    if not rows:
        raise ValueError('Empty training data')
    sizes = np.array([len(r['x']) for r in rows])
    offsets = np.r_[0, np.cumsum(sizes)]
    x = np.concatenate([r['x'] for r in rows])
    positive = offsets[:-1] + np.array([r['y'] for r in rows])
    def objective(w):
        logits = x @ w
        maxima = np.maximum.reduceat(logits, offsets[:-1])
        exp = np.exp(logits - np.repeat(maxima, sizes))
        sums = np.add.reduceat(exp, offsets[:-1])
        loss = np.mean(np.log(sums)+maxima-logits[positive]) + regularization*np.dot(w,w)/2
        grad_logits = exp / np.repeat(sums, sizes)
        grad_logits[positive] -= 1
        grad = x.T @ grad_logits / len(rows) + regularization*w
        return float(loss), grad
    result = minimize(objective, np.zeros(len(FEATURES)), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=300, ftol=1e-12, gtol=1e-8))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Fit did not converge: '+str(result.message))
    return result.x, dict(loss=float(result.fun), iterations=int(result.nit), queries=len(rows))


def evaluate(rows, weights):
    result = dict(queries=len(rows), learned_correct=0, current_correct=0, initial_correct=0,
                  learned_repairs=0, learned_breaks=0, nll=0.)
    for row in rows:
        logits = row['x'] @ weights
        choice = int(np.argmax(logits)); y = row['y']; truth = int(row['parents'][y])
        correct = choice == y; current = row['current'] == truth
        result['learned_correct'] += int(correct)
        result['current_correct'] += int(current)
        result['initial_correct'] += int(row['initial'] == truth)
        result['learned_repairs'] += int(correct and not current)
        result['learned_breaks'] += int(not correct and current)
        result['nll'] += float(np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[y])
    result['nll'] /= max(1,len(rows))
    return result
