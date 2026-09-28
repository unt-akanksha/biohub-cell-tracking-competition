"""Joint continuation, fork, birth and death choices with partial-label learning.

Options are (parent_index, child_index, second_child_index), with -1 denoting
absence. One option is chosen for each parent; each child has one explanation.
No detections are added, removed, relocated, or scored by a node-count target.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class EventSolveError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def problem(options, nparents, nchildren):
    options = np.asarray(options)
    if (options.ndim != 2 or options.shape[1] != 3 or options.dtype.kind not in 'iu'
            or nparents <= 0 or nchildren <= 0 or not len(options)):
        raise ValueError('Invalid event option schema')
    options = options.astype(np.int64)
    if len({tuple(row) for row in options}) != len(options):
        raise ValueError('Duplicate event options')
    rows, cols = [], []
    for index, (parent, first, second) in enumerate(options):
        if (not -1 <= parent < nparents or not -1 <= first < nchildren or not -1 <= second < nchildren
                or (first < 0 and second >= 0) or (second >= 0 and first >= second)
                or (parent < 0 and (first < 0 or second >= 0))):
            raise ValueError('Invalid death, birth, continuation or canonical two-child fork')
        if parent >= 0:
            rows.append(parent); cols.append(index)
        for child in (first, second):
            if child >= 0:
                rows.append(nparents + child); cols.append(index)
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(nparents + nchildren, len(options))).tocsc()
    if np.any(np.asarray(matrix.sum(axis=1)).ravel() == 0):
        raise ValueError('At least one cell has no event option')
    return dict(options=options, matrix=matrix, nparents=nparents, nchildren=nchildren)


def validate_choice(case, choice):
    choice = np.asarray(choice)
    if (choice.shape != (len(case['options']),) or not np.isin(choice, (0, 1)).all()
            or not np.array_equal(case['matrix'] @ choice, np.ones(case['matrix'].shape[0]))):
        raise ValueError('Event choices do not form a feasible lineage assignment')


def solve(case, scores, *, allowed=None, time_limit=5.):
    scores = np.asarray(scores, np.float64)
    if scores.shape != (len(case['options']),) or not np.isfinite(scores).all():
        raise ValueError('Invalid event scores')
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('Positive finite solver deadline required')
    if allowed is None:
        allowed = np.ones(len(scores), bool)
    allowed = np.asarray(allowed)
    if allowed.shape != scores.shape or allowed.dtype.kind != 'b':
        raise ValueError('Invalid allowed-option mask')
    result = milp(-scores, integrality=np.ones(len(scores)),
                  bounds=Bounds(np.zeros(len(scores)), allowed.astype(float)),
                  constraints=LinearConstraint(case['matrix'], 1., 1.),
                  options=dict(time_limit=float(time_limit), mip_rel_gap=0.))
    if result.status != 0 or not result.success or result.x is None:
        raise EventSolveError('Optimal event assignment unavailable, solver status ' + str(result.status), status=result.status)
    if np.asarray(result.x).shape != scores.shape or not np.isfinite(result.x).all():
        raise EventSolveError('Nonfinite or malformed solver output')
    rounded = np.rint(result.x).astype(np.int8)
    if not np.allclose(result.x, rounded, atol=1e-6, rtol=0):
        raise EventSolveError('Nonintegral solver output')
    validate_choice(case, rounded)
    if np.any(rounded[~allowed]):
        raise EventSolveError('Solver selected a forbidden option')
    return rounded


def prepare(case, features, target_parents, safe_alternatives):
    """Known child->parent constraints; unknown cells and close rivals add no loss.

    target_parents uses -1 for unknown. A missing annotated parent is NOT
    relabeled as a biological birth. safe_alternatives has two entries per
    option corresponding to its children; only known children use that mask.
    """
    x = np.asarray(features, np.float64)
    targets = np.asarray(target_parents)
    safe = np.asarray(safe_alternatives)
    if x.ndim != 2 or x.shape[0] != len(case['options']) or not np.isfinite(x).all():
        raise ValueError('Invalid event features')
    if (targets.shape != (case['nchildren'],) or targets.dtype.kind not in 'iu'
            or np.any(targets < -1) or np.any(targets >= case['nparents'])):
        raise ValueError('Targets must be reachable parent indices or unknown, never assumed births')
    if safe.shape != (len(case['options']), 2) or safe.dtype.kind != 'b':
        raise ValueError('Invalid safe-alternative mask')
    known = targets >= 0
    if not known.any():
        return None, 'no_known_parent_constraints'
    allowed = np.ones(len(case['options']), bool)
    margin = np.zeros(len(case['options']), np.float64)
    for index, (parent, first, second) in enumerate(case['options']):
        for slot, child in enumerate((first, second)):
            if child < 0 or not known[child] or parent == targets[child]:
                continue
            allowed[index] = False
            # A true parent's available child assigned to birth is a known
            # missed link. Unknown daughters are never false-positive labels.
            margin[index] += float(parent < 0 or safe[index, slot])
    try:
        solve(case, np.zeros(len(allowed)), allowed=allowed)
    except EventSolveError as error:
        if error.status != 2:
            raise
        return None, 'incompatible_partial_constraints'
    return dict(problem=case, x=x, allowed=allowed, margin=margin, n_constraints=int(known.sum())), 'prepared'


def hinge(case, weights, *, time_limit=5.):
    weights = np.asarray(weights, np.float64)
    scores = case['x'] @ weights
    predicted = solve(case['problem'], scores + case['margin'], time_limit=time_limit)
    latent = solve(case['problem'], scores, allowed=case['allowed'], time_limit=time_limit)
    count = case['n_constraints']
    value = float(((scores + case['margin']) @ predicted - scores @ latent) / count)
    gradient = case['x'].T @ (predicted.astype(float) - latent) / count
    if not np.isfinite(value) or value < -1e-7 or not np.isfinite(gradient).all():
        raise ValueError('Invalid partial-label event hinge')
    return max(0., value), gradient


def infer(case, scores, incumbent, *, time_limit=5.):
    """Exact solver only; a timeout preserves the complete incumbent assignment."""
    validate_choice(case, incumbent)
    try:
        chosen = solve(case, scores, time_limit=time_limit)
    except EventSolveError as error:
        return np.asarray(incumbent).copy(), dict(changed=False, fallback=True, reason=str(error))
    improvement = float(np.asarray(scores) @ (chosen.astype(float) - incumbent))
    if improvement <= 1e-9:
        return np.asarray(incumbent).copy(), dict(changed=False, fallback=False, gain=improvement)
    return chosen, dict(changed=not np.array_equal(chosen, incumbent), fallback=False, gain=improvement)
