"""Image-only downstream parent groups and sparse source-label coverage."""
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree
from research.native_correspondence_data_v2 import match_queries
from research.trajectory_disagreement_data_v1 import positions, adjacency


def candidates(initial, final):
    """Top 16 observed parents within 20um, union existing stage choices."""
    initial_nodes = {int(k): v for k, v in initial['nodes'].items()}
    nodes = {int(k): v for k, v in final['nodes'].items()}
    observed = sorted(initial_nodes.keys() & nodes.keys())
    pos = positions(nodes)
    by_t = defaultdict(list)
    for i in observed:
        by_t[int(nodes[i]['t'])].append(i)
    initial_in, _ = adjacency(initial['edges'])
    final_in, _ = adjacency(final['edges'])
    children, parents, offsets, current, neural = [], [], [0], [], []
    for t in sorted(by_t):
        parent_ids = by_t.get(t - 1, [])
        if not parent_ids:
            continue
        coords = np.stack([pos[i] for i in parent_ids])
        tree = cKDTree(coords)
        for child in by_t[t]:
            # Leave nonconsecutive gap edges and nonshared/synthetic parents alone.
            selected = final_in[child]
            if len(selected) > 1:
                raise ValueError('Multiple current parents')
            if selected and selected[0] not in parent_ids:
                continue
            nearby = tree.query_ball_point(pos[child], 20.)
            nearby.sort(key=lambda j: (float(np.linalg.norm(coords[j] - pos[child])), parent_ids[j]))
            proposed = {parent_ids[j] for j in nearby[:16]}
            old = [p for p in initial_in[child] if p in parent_ids]
            proposed.update(selected)
            proposed.update(old)
            if not proposed:
                continue
            children.append(child)
            parents.extend(sorted(proposed))
            offsets.append(len(parents))
            current.append(selected[0] if selected else -1)
            neural.append(old[0] if len(old) == 1 else -1)
    result = dict(children=np.asarray(children, np.int64), parents=np.asarray(parents, np.int64),
                  offsets=np.asarray(offsets, np.int64), current=np.asarray(current, np.int64),
                  neural=np.asarray(neural, np.int64))
    assert len(result['offsets']) == len(children) + 1
    return result


def label(groups, final, truth_nodes, truth_edges):
    nodes = {int(k): v for k, v in final['nodes'].items()}
    truth_nodes = {int(k): v for k, v in truth_nodes.items()}
    pos, gt_pos = positions(nodes), positions(truth_nodes)
    predicted_by_t, truth_by_t = defaultdict(list), defaultdict(list)
    for i, n in sorted(nodes.items()):
        predicted_by_t[int(n['t'])].append(i)
    for i, n in sorted(truth_nodes.items()):
        truth_by_t[int(n['t'])].append(i)
    gt_to_det, det_to_gt = {}, {}
    for t, pred_ids in predicted_by_t.items():
        truth_ids = truth_by_t[t]
        if not truth_ids:
            continue
        matched = match_queries(np.stack([gt_pos[i] for i in truth_ids]), np.stack([pos[i] for i in pred_ids]))
        for a, b in matched.items():
            gt_to_det[truth_ids[a]] = pred_ids[b]
            det_to_gt[pred_ids[b]] = truth_ids[a]
    gt_in = defaultdict(list)
    for a, b in truth_edges:
        gt_in[int(b)].append(int(a))
    _, outgoing = adjacency(final['edges'])
    target = np.full(len(groups['children']), -1, np.int64)
    safe = np.zeros(len(groups['parents']), bool)
    counts = Counter()
    for i, child in enumerate(groups['children']):
        gchild = det_to_gt.get(int(child))
        if gchild is None:
            counts['unknown_child'] += 1
            continue
        if len(gt_in[gchild]) != 1:
            counts['no_unique_annotated_parent'] += 1
            continue
        gparent = gt_in[gchild][0]
        if int(truth_nodes[gparent]['t']) != int(truth_nodes[gchild]['t']) - 1:
            counts['nonconsecutive_annotation'] += 1
            continue
        a, b = groups['offsets'][i:i+2]
        choices = groups['parents'][a:b]
        distances = np.array([np.linalg.norm(pos[int(p)] - gt_pos[gparent]) for p in choices])
        expected = gt_to_det.get(gparent)
        good = np.flatnonzero(choices == expected) if expected is not None else np.array([], int)
        if len(good) and distances[good[0]] <= 3.25:
            target[i] = int(expected)
            safe[a:b] = distances > 7.
            safe[a+good[0]] = True
            counts['positive_parent_present'] += 1
            counts['safe_negative_candidates'] += int((distances > 7.).sum())
            counts['positive_with_safe_alternative'] += int((distances > 7.).any())
            selected = int(groups['current'][i])
            if selected == expected:
                counts['current_correct'] += 1
            elif selected < 0 or np.linalg.norm(pos[selected] - gt_pos[gparent]) > 7.:
                counts['current_definitely_wrong_or_absent'] += 1
                counts['repair_parent_' + ('free' if not outgoing[expected] else 'occupied')] += 1
            else:
                counts['current_ambiguous'] += 1
            initial = int(groups['neural'][i])
            counts['initial_correct'] += int(initial == expected)
        elif np.all(distances > 7.):
            target[i] = -2
            safe[a:b] = True
            counts['known_parent_absent'] += 1
        else:
            counts['ambiguous_parent'] += 1
    counts['groups'] = len(target)
    counts['candidate_edges'] = len(safe)
    return target, safe, dict(counts)
