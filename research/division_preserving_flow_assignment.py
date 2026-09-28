"""Joint ordinary-link assignment with the existing image-flow forks locked.

No labels, count targets, new divisions, node deletion, or tuned constants.
The null and Gaussian costs are the existing source-trained motion settings.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from research.backward_flow_linking import link_backward_flow
from research.independent_motion_prior import SCALE, VARIANCE, NULL_LOGIT


def ordinary_assignment(costs):
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Finite nonnegative source-by-child costs required')
    ns, nt = costs.shape
    if max(ns, nt) > 2048:
        raise ValueError('Assignment guard exceeded; no truncation')
    if not ns or not nt:
        return []
    null_cost = -NULL_LOGIT
    matrix = np.full((nt, ns + nt), np.inf)
    matrix[:, :ns] = np.where(costs.T < null_cost, costs.T, np.inf)
    matrix[np.arange(nt), ns + np.arange(nt)] = null_cost
    rows, columns = linear_sum_assignment(matrix)
    return sorted((int(s), int(t)) for t, s in zip(rows, columns) if s < ns)


def forks(edges):
    children = defaultdict(list)
    for s, d in edges:
        children[int(s)].append(int(d))
    return {s: tuple(sorted(ds)) for s, ds in children.items() if len(ds) == 2}


def validate_edges(coords, edges, expected_forks):
    if len(set(edges)) != len(edges):
        raise ValueError('Duplicate edge')
    parents, children = Counter(), Counter()
    for s, d in edges:
        if (int(s) != s or int(d) != d or not 0 <= s < len(coords)
                or not 0 <= d < len(coords) or coords[d, 0] != coords[s, 0] + 1):
            raise ValueError('Actual next-frame node indices required')
        parents[d] += 1
        children[s] += 1
    if max(parents.values(), default=0) > 1 or max(children.values(), default=0) > 2:
        raise ValueError('Invalid tracking topology')
    if forks(edges) != expected_forks:
        raise ValueError('Existing division edges must be preserved exactly; no new forks')


def link(coords, backward_um):
    points = np.asarray(coords, dtype=np.float64)
    flow = np.asarray(backward_um, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 4 or flow.shape != (len(points), 3)
            or not np.isfinite(points).all() or not np.isfinite(flow).all()
            or (points < 0).any() or np.any(points[:, 0] != np.floor(points[:, 0]))
            or np.any(flow[points[:, 0] == 0] != 0)):
        raise ValueError('Finite unchanged TZYX nodes and aligned backward flow required')
    original = [(s, d) for s, d, probability in link_backward_flow(points, flow)]
    locked = forks(original)
    edges = [(s, d) for s, ds in locked.items() for d in ds]
    for t in sorted(set(points[:, 0].astype(int))):
        source = np.flatnonzero(points[:, 0] == t)
        target = np.flatnonzero(points[:, 0] == t + 1)
        if max(len(source), len(target)) > 2048:
            raise ValueError('Assignment guard exceeded; no truncation')
        locked_children = {d for s in source if s in locked for d in locked[s]}
        source = np.asarray([s for s in source if s not in locked], dtype=np.int64)
        target = np.asarray([d for d in target if d not in locked_children], dtype=np.int64)
        delta = points[source, None, 1:] * SCALE - (points[None, target, 1:] * SCALE + flow[None, target])
        costs = .5 * np.sum(delta ** 2 / VARIANCE, axis=-1)
        edges.extend((int(source[s]), int(target[d])) for s, d in ordinary_assignment(costs))
    edges = sorted(edges)
    validate_edges(points, edges, locked)
    return edges, dict(locked_divisions=len(locked), original_edges=len(original),
                      candidate_edges=len(edges), added_edges=len(set(edges)-set(original)),
                      removed_edges=len(set(original)-set(edges)), all_division_edges_preserved=True)
