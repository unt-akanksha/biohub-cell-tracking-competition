from copy import deepcopy
from research.focus_edge_consensus import repair, mutual_matches


def graph(edges=()):
    return {'nodes': {str(i): dict(node_id=i, t=i, z=1., y=2., x=3.) for i in range(3)},
            'edges': [dict(source_id=s, target_id=t) for s, t in edges]}


def test_adds_supported_break_without_mutating_inputs():
    base, external = graph([(0, 1)]), graph([(0, 1), (1, 2)])
    before = deepcopy(base)
    result, receipt = repair(base, external)
    assert base == before and result['nodes'] == before['nodes']
    assert result['edges'] == external['edges']
    assert receipt['added_edges'] == 1


def test_does_not_duplicate_existing_edges():
    base = graph([(0, 1), (1, 2)])
    result, receipt = repair(base, base)
    assert result == base and receipt['added_edges'] == 0


def test_ambiguous_and_distant_matches_abstain():
    base = {int(k): v for k, v in graph()['nodes'].items()}
    external = deepcopy(base)
    external[10] = dict(external[0], node_id=10)
    external[1]['x'] = 100.
    matches = mutual_matches(base, external)
    assert matches == {2: 2}


def test_division_source_is_not_used_as_continuation_support():
    external = graph([(0, 1), (0, 2)])
    external['nodes']['2']['t'] = 1
    base = deepcopy(external)
    base['edges'] = []
    result, receipt = repair(base, external)
    assert not result['edges'] and receipt['added_edges'] == 0


def test_existing_successor_or_predecessor_prevents_reassignment():
    base = graph([(0, 1)])
    base['nodes']['2'] = dict(node_id=2, t=1, z=10., y=2., x=3.)
    external = deepcopy(base)
    external['edges'] = [dict(source_id=0, target_id=2)]
    result, receipt = repair(base, external)
    assert result == base and receipt['added_edges'] == 0
    base['edges'] = [dict(source_id=0, target_id=1)]
    base['nodes']['2']['t'] = 0
    external = deepcopy(base)
    external['edges'] = [dict(source_id=2, target_id=1)]
    result, receipt = repair(base, external)
    assert result == base and receipt['added_edges'] == 0


def test_inherited_negative_coordinate_is_preserved_not_hidden():
    base, external = graph(), graph([(0, 1)])
    base['nodes']['0']['z'] = -0.001
    result, receipt = repair(base, external)
    assert result['nodes'] == base['nodes']
    assert result['nodes']['0']['z'] == -0.001
    assert receipt['coordinate_changes'] == 0
