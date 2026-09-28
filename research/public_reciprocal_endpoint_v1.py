"""Image/model-supported adjacent track reconnection; no truth or node edits."""
import copy
from collections import defaultdict

import numpy as np

SCALE = np.asarray((1.625, .40625, .40625))


def reconnect(final, coords, edges, original_detector_ids):
    coords = np.asarray(coords)
    edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    original = set(map(int, original_detector_ids))
    nodes = {int(k): v for k, v in final['nodes'].items()}
    if (coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all()
            or not np.equal(coords, np.round(coords)).all()
            or (coords < 0).any()
            or (coords >= np.asarray((100, 64, 256, 256))).any()):
        raise ValueError('Invalid native-voxel detector coordinates')
    if any(i < 0 or i >= len(coords) for i in original):
        raise ValueError('Invalid genuine detector ID')
    incoming, outgoing = defaultdict(list), defaultdict(list)
    existing = set()
    for i, n in nodes.items():
        if int(n['node_id']) != i or not np.isfinite([n[k] for k in ('t', 'z', 'y', 'x')]).all():
            raise ValueError('Invalid final node')
        if i in original and int(n['t']) != coords[i, 0]:
            raise ValueError('Raw/final detector time mismatch')
    for e in final['edges']:
        a, b = int(e['source_id']), int(e['target_id'])
        if a not in nodes or b not in nodes or nodes[b]['t'] != nodes[a]['t'] + 1:
            raise ValueError('Invalid original temporal edge')
        if (a, b) in existing:
            raise ValueError('Duplicate original edge')
        existing.add((a, b)); incoming[b].append(a); outgoing[a].append(b)
    if max(map(len, incoming.values()), default=0) > 1 or max(map(len, outgoing.values()), default=0) > 2:
        raise ValueError('Invalid original lineage degree')
    forward, backward = defaultdict(dict), defaultdict(dict)
    for a, b, p, d in edges:
        if not np.isfinite([a, b, p, d]).all() or a != int(a) or b != int(b):
            raise ValueError('Invalid raw edge')
        a, b = int(a), int(b)
        if not (0 <= a < len(coords) and 0 <= b < len(coords) and 0 <= p <= 1):
            raise ValueError('Invalid raw edge range')
        if coords[b, 0] != coords[a, 0] + 1:
            raise ValueError('Raw edge must be adjacent in time')
        forward[a][b] = max(forward[a].get(b, 0.), p)
        backward[b][a] = max(backward[b].get(a, 0.), p)

    def best(table):
        result = {}
        for i, choices in table.items():
            maximum = max(choices.values())
            winners = [j for j, p in choices.items() if p == maximum]
            if maximum >= .88 and len(winners) == 1:
                result[i] = (winners[0], maximum)
        return result

    fb, bb = best(forward), best(backward)

    def point(i):
        return np.asarray([nodes[i][k] for k in ('z', 'y', 'x')]) * SCALE

    def history(i, reverse):
        chain = [i]
        lookup, reciprocal = (incoming, outgoing) if reverse else (outgoing, incoming)
        for _ in range(2):
            choices = lookup[chain[-1]]
            if len(choices) != 1 or len(reciprocal[choices[0]]) != 1:
                return None
            chain.append(choices[0])
        # Extrapolate one step away from the existing three-node segment.
        return (4 * point(chain[0]) + point(chain[1]) - 2 * point(chain[2])) / 3

    records = []
    counts = dict(reciprocal_raw=0, genuine_endpoint_pairs=0,
                  distance_pass=0, two_sided_history=0)
    for a in sorted(fb):
        b, p = fb[a]
        if bb.get(b, (None,))[0] != a:
            continue
        counts['reciprocal_raw'] += 1
        if a not in nodes or b not in nodes or a not in original or b not in original:
            continue
        if outgoing[a] or incoming[b]:
            continue
        counts['genuine_endpoint_pairs'] += 1
        distance = float(np.linalg.norm(point(b) - point(a)))
        if distance > 6:
            continue
        counts['distance_pass'] += 1
        future_a, past_b = history(a, True), history(b, False)
        if future_a is None or past_b is None:
            continue
        counts['two_sided_history'] += 1
        forward_error = float(np.linalg.norm(future_a - point(b)))
        backward_error = float(np.linalg.norm(past_b - point(a)))
        if max(forward_error, backward_error) > 3:
            continue
        records.append(dict(source_id=a, target_id=b, probability=float(p),
                            distance_um=distance, forward_error_um=forward_error,
                            backward_error_um=backward_error))
    if len({r['source_id'] for r in records}) != len(records) or len({r['target_id'] for r in records}) != len(records):
        raise ValueError('Reciprocal endpoints must be disjoint')
    result = copy.deepcopy(final)
    result['edges'].extend({k: r[k] for k in ('source_id', 'target_id')} for r in records)
    return result, dict(**counts, added_nodes=0, added_edges=len(records), links=records,
                        ground_truth_used=False, existing_output_unchanged=True,
                        authorized_for_submission=False)
