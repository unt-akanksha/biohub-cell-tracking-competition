"""Sparse physical supervision: unknown cells are never background/birth labels."""
from collections import Counter, defaultdict
import numpy as np

from research.trajectory_disagreement_data_v1 import positions


def label(groups, nodes, truth_nodes, truth_edges, det_to_gt):
    nodes = {int(k): v for k, v in nodes.items()}
    truth = {int(k): v for k, v in truth_nodes.items()}
    matched = {int(k): int(v) for k, v in det_to_gt.items()}
    if len(set(matched.values())) != len(matched):
        raise ValueError('Physical matching must be one-to-one')
    pos, gt_pos = positions(nodes), positions(truth)
    for predicted, known in matched.items():
        if (predicted not in nodes or known not in truth
                or nodes[predicted]['t'] != truth[known]['t']
                or np.linalg.norm(pos[predicted] - gt_pos[known]) > 7. + 1e-6):
            raise ValueError('Invalid physical source match')
    inverse = {known: predicted for predicted, known in matched.items()}
    incoming = defaultdict(list)
    for parent, child in truth_edges:
        incoming[int(child)].append(int(parent))
    targets = np.full(len(groups['children']), -1, np.int64)
    safe = np.zeros(len(groups['parents']), bool)
    counts = Counter()
    for index, child in enumerate(groups['children']):
        known = matched.get(int(child))
        if known is None:
            counts['unknown_child'] += 1
            continue
        if len(incoming[known]) != 1:
            counts['no_unique_annotated_parent'] += 1
            continue
        parent = incoming[known][0]
        if truth[known]['t'] - truth[parent]['t'] != 1:
            counts['nonconsecutive_annotation'] += 1
            continue
        a, b = groups['offsets'][index:index+2]
        choices = groups['parents'][a:b]
        expected = inverse.get(parent)
        if expected is None or expected not in choices:
            counts['matched_parent_missing_or_outside_candidates'] += 1
            continue
        if np.count_nonzero(choices == expected) != 1:
            raise ValueError('Duplicate positive candidate')
        targets[index] = expected
        distances = np.array([np.linalg.norm(pos[int(p)] - gt_pos[parent]) for p in choices])
        safe[a:b] = distances > 7.
        safe[a + int(np.flatnonzero(choices == expected)[0])] = True
        counts['known_parent_constraints'] += 1
        counts['safe_wrong_parent_options'] += int((distances > 7.).sum())
        counts['current_correct'] += int(groups['current'][index] == expected)
    return targets, safe, dict(counts)


def for_problem(problem, group_targets, group_safe):
    """Map known physical IDs into an editable event problem without fake births."""
    raw = group_targets[problem['group_indices']]
    parents = {int(p): i for i, p in enumerate(problem['parents'])}
    targets = np.array([parents.get(int(p), -1) if p >= 0 else -1 for p in raw], np.int64)
    mask = problem['edge_indices'] >= 0
    safe = np.zeros_like(problem['edge_indices'], bool)
    safe[mask] = group_safe[problem['edge_indices'][mask]]
    return targets, safe, dict(known_constraints_retained=int((targets >= 0).sum()),
                               known_constraints_outside_editable_scope=int(((raw >= 0) & (targets < 0)).sum()))
