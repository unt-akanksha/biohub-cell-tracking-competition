"""Prediction-only inventory of existing forks locked by the v1 event head."""
from research.trajectory_disagreement_data_v1 import adjacency


def inventory(initial, graph):
    nodes = {int(k): v for k, v in graph['nodes'].items()}
    observed = set(map(int, initial['nodes'])) & nodes.keys()
    incoming, outgoing = adjacency(graph['edges'])
    hard_protected = set()
    for edge in graph['edges']:
        a, b = int(edge['source_id']), int(edge['target_id'])
        if a not in observed or b not in observed or nodes[b]['t'] - nodes[a]['t'] != 1:
            hard_protected.update((a, b))
    forks = {}
    for parent, children in outgoing.items():
        if len(children) < 2:
            continue
        if len(children) != 2 or len(set(children)) != 2:
            raise ValueError('Invalid baseline fork degree')
        family = {parent, *children}
        eligible = family <= observed and not family.intersection(hard_protected)
        forks[parent] = dict(children=sorted(children), observed_consecutive_scope=bool(eligible),
            locked_by_current_event_head=True,
            mother_has_single_consecutive_history=len(incoming[parent]) == 1 and
                nodes[parent]['t'] - nodes[incoming[parent][0]]['t'] == 1,
            both_daughters_have_single_consecutive_future=all(len(outgoing[c]) == 1 and
                nodes[outgoing[c][0]]['t'] - nodes[c]['t'] == 1 for c in children))
    return forks
