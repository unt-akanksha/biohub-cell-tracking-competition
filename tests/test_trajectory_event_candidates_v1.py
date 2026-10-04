import numpy as np

from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_assignment_v1 import validate_choice


def graph():
    return dict(nodes={str(i): dict(node_id=i, t=int(i >= 2), z=0, y=0, x=i) for i in range(4)},
                edges=[dict(source_id=0, target_id=2)])


def test_new_fork_available_while_incumbent_and_birth_remain_feasible():
    initial = graph()
    groups = candidates(initial, initial)
    case, = list(frames(initial, initial, groups))
    validate_choice(case, case['incumbent'])
    assert (0,0,1) in {tuple(row) for row in case['options']}
    assert (-1,1,-1) in {tuple(row) for row in case['options']}
    assert len(case['children']) == 2


def test_existing_division_is_excluded_not_silently_reassigned():
    initial = graph()
    initial['edges'].append(dict(source_id=0, target_id=3))
    assert list(frames(initial, initial, candidates(initial, initial))) == []


def test_synthetic_boundary_edges_stay_outside_editable_scope():
    initial = graph()
    final = graph()
    final['nodes']['4'] = dict(node_id=4, t=1, z=0, y=0, x=4)
    final['edges'].append(dict(source_id=1, target_id=4))
    case, = list(frames(initial, final, candidates(initial, final)))
    assert 1 not in case['parents'] and 4 not in case['children']
    assert np.all(case['edge_indices'] < len(candidates(initial, final)['parents']))


def test_uniform_fork_expansion_is_superset_and_restores_distant_daughter_pair():
    from research.trajectory_event_fast_training_v1 import prepare
    initial = dict(
        nodes={str(i): dict(node_id=i, t=int(i > 0), z=0, y=0, x=i) for i in range(11)},
        edges=[dict(source_id=0, target_id=1)],
    )
    groups = candidates(initial, initial)
    narrow, = list(frames(initial, initial, groups, max_fork_children=8))
    wide, = list(frames(initial, initial, groups, max_fork_children=16))
    assert set(map(tuple, narrow['options'])) < set(map(tuple, wide['options']))
    assert np.array_equal(narrow['parents'], wide['parents'])
    assert np.array_equal(narrow['children'], wide['children'])
    # The grammar expands before labels are supplied. Unknown children remain
    # unconstrained; neither labels nor scoring code inserts a custom option.
    targets = np.full(wide['nchildren'], -1, dtype=np.int64)
    targets[np.isin(wide['children'], [9, 10])] = 0
    assert np.count_nonzero(targets >= 0) == 2
    reasons = []
    for case in (narrow, wide):
        validate_choice(case, case['incumbent'])
        _, reason = prepare(case, np.zeros((len(case['options']), 1)), targets,
                            np.zeros((len(case['options']), 2), dtype=bool))
        reasons.append(reason)
    assert reasons == ['incompatible_partial_constraints', 'prepared']
    assert initial['edges'] == [dict(source_id=0, target_id=1)]
