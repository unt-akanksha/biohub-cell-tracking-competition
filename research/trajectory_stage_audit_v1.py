"""Diagnostic edge provenance; never generate or modify prediction graphs."""
from collections import Counter


def attribute(nodes, edges, truth_edges, initial_edges):
    truth_edges = set(map(tuple, truth_edges))
    initial_edges = set(map(tuple, initial_edges))
    matched_gt = {r['matched'] for r in nodes.values() if r['matched'] is not None and r['matched'] >= 0}
    recovered = set()
    counts = Counter()
    for edge in edges:
        a, b = nodes[edge['source']], nodes[edge['target']]
        origin = 'in_initial_graph' if (a['original'], b['original']) in initial_edges else 'introduced_after_initial_graph'
        if edge['matched']:
            pair = (a['matched'], b['matched'])
            if pair not in truth_edges or pair in recovered:
                raise ValueError('Nonunique or invalid official matched edge')
            recovered.add(pair)
            counts[origin + '_tp'] += 1
        elif edge['valid']:
            counts[origin + '_fp'] += 1
    missing = truth_edges - recovered
    for a, b in missing:
        reason = 'fn_endpoint_unmatched' if a not in matched_gt or b not in matched_gt else 'fn_both_endpoints_matched'
        counts[reason] += 1
    return dict(counts= dict(counts), recovered_gt_edges=sorted(recovered), missing_gt_edges=sorted(missing))
