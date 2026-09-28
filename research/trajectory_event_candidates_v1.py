"""Label-free bounded fork options over existing observed predictor candidates."""
from collections import defaultdict
from itertools import combinations
import numpy as np

from research.trajectory_disagreement_data_v1 import adjacency, positions
from research.trajectory_event_assignment_v1 import problem, validate_choice


def frames(initial, final, groups, *, max_fork_children=8, selected_times=None):
    if not isinstance(max_fork_children, int) or not 2 <= max_fork_children <= 16:
        raise ValueError('Fork candidate budget must be between two and sixteen')
    nodes = {int(k): v for k, v in final['nodes'].items()}
    observed = set(map(int, initial['nodes'])) & set(nodes)
    incoming, outgoing = adjacency(final['edges'])
    protected = set()
    for edge in final['edges']:
        a, b = int(edge['source_id']), int(edge['target_id'])
        if (a not in observed or b not in observed or len(outgoing[a]) == 2
                or nodes[b]['t'] - nodes[a]['t'] != 1):
            protected.update((a, b))
    by_t = defaultdict(list)
    group_index = {int(c): i for i, c in enumerate(groups['children'])}
    if len(group_index) != len(groups['children']):
        raise ValueError('Duplicate child groups')
    for i, child in enumerate(groups['children']):
        if int(child) in observed and int(child) not in protected:
            by_t[int(nodes[int(child)]['t'])].append(i)
    pos = positions(nodes)
    for t, indices in sorted(by_t.items()):
        if selected_times is not None and t not in selected_times:
            continue
        candidate_map = {}
        for i in indices:
            child = int(groups['children'][i])
            a, b = groups['offsets'][i:i+2]
            candidate_map[child] = {int(groups['parents'][j]): int(j) for j in range(a, b)
                                    if int(groups['parents'][j]) in observed
                                    and int(groups['parents'][j]) not in protected
                                    and int(nodes[int(groups['parents'][j])]['t']) == t - 1}
        children = set(candidate_map)
        parents = {p for choices in candidate_map.values() for p in choices}
        # Preserve every edge crossing the scope boundary; close removals until
        # all currently selected edges inside this bipartite problem are free.
        while True:
            new_parents = {p for p in parents if all(c in children for c in outgoing[p])}
            new_children = {c for c in children if all(p in new_parents for p in incoming[c])}
            if new_parents == parents and new_children == children:
                break
            parents, children = new_parents, new_children
        if not parents or not children:
            continue
        parents, children = sorted(parents), sorted(children)
        pi, ci = {p: i for i, p in enumerate(parents)}, {c: i for i, c in enumerate(children)}
        options, edge_indices = [], []
        def add(parent, first=-1, second=-1, first_edge=-1, second_edge=-1):
            options.append((parent, first, second)); edge_indices.append((first_edge, second_edge))
        for parent in parents:
            add(pi[parent])
            choices = sorted(c for c in children if parent in candidate_map[c])
            for child in choices:
                add(pi[parent], ci[child], first_edge=candidate_map[child][parent])
            closest = sorted(choices, key=lambda c: (float(np.linalg.norm(pos[c] - pos[parent])), c))[:max_fork_children]
            for a, b in combinations(sorted(closest), 2):
                add(pi[parent], ci[a], ci[b], candidate_map[a][parent], candidate_map[b][parent])
        for child in children:
            add(-1, ci[child])
        result = problem(options, len(parents), len(children))
        index = {tuple(row): i for i, row in enumerate(options)}
        incumbent = np.zeros(len(options), np.int8)
        for parent in parents:
            kids = sorted(outgoing[parent])
            if len(kids) > 1:
                raise ValueError('Existing division was not protected')
            option = (pi[parent], ci[kids[0]] if kids else -1, -1)
            if option not in index:
                raise ValueError('Current graph is absent from candidate options')
            incumbent[index[option]] = 1
        for child in children:
            if not incoming[child]:
                incumbent[index[(-1, ci[child], -1)]] = 1
        validate_choice(result, incumbent)
        result.update(t=t, parents=np.asarray(parents), children=np.asarray(children),
                      edge_indices=np.asarray(edge_indices), incumbent=incumbent,
                      group_indices=np.array([group_index[c] for c in children]),
                      max_fork_children=max_fork_children, existing_division_and_synthetic_incidence_protected=True)
        yield result
