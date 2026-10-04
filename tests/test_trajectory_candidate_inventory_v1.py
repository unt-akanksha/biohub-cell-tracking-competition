import numpy as np
from research.trajectory_candidate_inventory_v1 import candidates, label


def node(t, x):
    return dict(t=t, z=0., y=0., x=x)


def test_candidate_set_unions_current_and_initial_beyond_radius():
    nodes = {0: node(0, 0), 1: node(0, 80), 2: node(1, 1)}
    initial = dict(nodes=nodes, edges=[dict(source_id=1, target_id=2)])
    final = dict(nodes=nodes, edges=[dict(source_id=0, target_id=2)])
    groups = candidates(initial, final)
    assert groups['children'].tolist() == [2]
    assert groups['parents'].tolist() == [0, 1]
    assert groups['current'].tolist() == [0]
    assert groups['neural'].tolist() == [1]


def test_supervision_known_repair_and_unknown_not_negative():
    nodes = {0: node(0, 0), 1: node(0, 40), 2: node(1, 1), 3: node(1, 80)}
    graph = dict(nodes=nodes, edges=[dict(source_id=1, target_id=2)])
    groups = candidates(graph, graph)
    target, safe, counts = label(groups, graph, {10: node(0, 0), 11: node(1, 1)}, [(10, 11)])
    assert target.tolist() == [0, -1]
    assert counts['current_definitely_wrong_or_absent'] == 1
    assert counts['repair_parent_free'] == 1
    assert not safe[groups['offsets'][1]:].any()


def test_close_alternative_is_not_negative():
    nodes = {0: node(0, 0), 1: node(0, 10), 2: node(1, 1)}
    graph = dict(nodes=nodes, edges=[dict(source_id=1, target_id=2)])
    target, safe, counts = label(candidates(graph, graph), graph,
                                 {10: node(0, 0), 11: node(1, 1)}, [(10, 11)])
    assert target.tolist() == [0] and safe.tolist() == [True, False]
    assert counts['current_ambiguous'] == 1
    assert counts['positive_with_safe_alternative'] == 0
