"""Append only reciprocal learned paths between genuine existing track endpoints."""
import copy
from collections import defaultdict
import numpy as np

SCALE = np.asarray((1.625, .40625, .40625))


def bridge(final, coords, edges):
    coords = np.asarray(coords)
    edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError('Invalid detector coordinates')
    if not np.equal(coords, np.round(coords)).all():
        raise ValueError('Expected integer original-voxel detector coordinates')
    if (coords < 0).any() or (coords >= np.asarray((100, 64, 256, 256))).any():
        raise ValueError('Detector point outside movie')
    nodes = {int(k): v for k, v in final['nodes'].items()}
    incoming, outgoing = defaultdict(int), defaultdict(int)
    frame_points = defaultdict(list)
    for ident, node in nodes.items():
        if int(node['node_id']) != ident:
            raise ValueError('Node identity mismatch')
        frame_points[int(node['t'])].append([node[k] for k in ('z', 'y', 'x')])
        if ident < len(coords) and int(node['t']) != int(coords[ident, 0]):
            raise ValueError('Detector identity/time mismatch')
    for edge in final['edges']:
        a, b = int(edge['source_id']), int(edge['target_id'])
        if a not in nodes or b not in nodes or nodes[b]['t'] != nodes[a]['t'] + 1:
            raise ValueError('Invalid parent graph')
        incoming[b] += 1
        outgoing[a] += 1
    if max(incoming.values(), default=0) > 1 or max(outgoing.values(), default=0) > 2:
        raise ValueError('Invalid parent lineage degrees')
    by_source, by_target = defaultdict(dict), defaultdict(dict)
    for a, b, probability, _ in edges:
        if not np.isfinite([a, b, probability]).all() or a != int(a) or b != int(b):
            raise ValueError('Invalid raw edge')
        a, b = int(a), int(b)
        if not (0 <= a < len(coords) and 0 <= b < len(coords)):
            raise ValueError('Dangling raw edge')
        if coords[b, 0] != coords[a, 0] + 1 or not 0 <= probability <= 1:
            raise ValueError('Invalid raw edge time/probability')
        by_source[a][b] = max(by_source[a].get(b, 0.), probability)
        by_target[b][a] = max(by_target[b].get(a, 0.), probability)

    def unique_best(table):
        best = {}
        for ident, choices in table.items():
            value = max(choices.values())
            winners = [j for j, p in choices.items() if p == value]
            if len(winners) == 1 and value >= .88:
                best[ident] = (winners[0], value)
        return best

    forward, backward = unique_best(by_source), unique_best(by_target)
    successor = {a: (b, p) for a, (b, p) in forward.items()
                 if backward.get(b, (None,))[0] == a}
    frame_points = {t: np.asarray(v, dtype=float) for t, v in frame_points.items()}

    def point(ident):
        if ident in nodes:
            return np.asarray([nodes[ident][k] for k in ('z', 'y', 'x')], dtype=float)
        return coords[ident, 1:].astype(float)

    proposals = []
    for start in sorted(nodes):
        if start >= len(coords) or outgoing[start]:
            continue
        chain, probabilities = [start], []
        for _ in range(3):
            pair = successor.get(chain[-1])
            if pair is None:
                break
            target, probability = pair
            if np.linalg.norm((point(target) - point(chain[-1])) * SCALE) > 6.:
                break
            chain.append(target)
            probabilities.append(probability)
            if target in nodes:
                if not incoming[target] and 1 <= len(chain) - 2 <= 2:
                    proposals.append((min(probabilities), float(np.mean(probabilities)), tuple(chain)))
                break
            nearby = frame_points.get(int(coords[target, 0]))
            if nearby is not None and np.any(np.linalg.norm((nearby - point(target)) * SCALE, axis=1) <= 3.):
                break
    proposals.sort(key=lambda row: (-row[0], -row[1], row[2]))
    result = copy.deepcopy(final)
    used, starts, ends, records = set(), set(), set(), []
    budget = min(120, max(0, int(round(len(nodes) * .012))))
    for minimum, mean, path in proposals:
        omitted = path[1:-1]
        if (used.intersection(omitted) or path[0] in starts or path[-1] in ends
                or len(used) + len(omitted) > budget):
            continue
        for ident in omitted:
            if str(ident) in result['nodes']:
                raise ValueError('Attempted overwrite')
            t, z, y, x = map(int, coords[ident])
            result['nodes'][str(ident)] = dict(node_id=ident, t=t, z=z, y=y, x=x)
        result['edges'].extend(dict(source_id=a, target_id=b) for a, b in zip(path, path[1:]))
        records.append(dict(node_ids=list(path), minimum_probability=minimum, mean_probability=mean))
        used.update(omitted)
        starts.add(path[0]); ends.add(path[-1])
    if any(result['nodes'][str(i)] != n for i, n in nodes.items()):
        raise ValueError('Existing node changed')
    return result, dict(eligible_paths=len(proposals), added_nodes=len(used),
        added_edges=len(result['edges']) - len(final['edges']), paths=records,
        node_budget=budget, existing_output_unchanged=True, ground_truth_used=False,
        authorized_for_submission=False)
