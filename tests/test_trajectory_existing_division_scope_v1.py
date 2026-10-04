from copy import deepcopy
import pytest
from research.trajectory_existing_division_scope_v1 import inventory


def graph():
    times = [0, 1, 2, 2, 3, 3]
    return dict(nodes={str(i): dict(t=t, z=0, y=0, x=i) for i, t in enumerate(times)},
        edges=[dict(source_id=a, target_id=b) for a, b in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5))])


def test_existing_observed_division_is_locked_but_research_eligible():
    g = graph()
    before = deepcopy(g)
    record = inventory(g, g)[1]
    assert record['observed_consecutive_scope'] and record['locked_by_current_event_head']
    assert record['mother_has_single_consecutive_history']
    assert record['both_daughters_have_single_consecutive_future']
    assert g == before


def test_synthetic_child_not_eligible():
    g = graph()
    initial = deepcopy(g)
    del initial['nodes']['3']
    assert not inventory(initial, g)[1]['observed_consecutive_scope']


def test_gap_incidence_on_daughter_blocks_reconciliation():
    g = graph()
    g['nodes']['4']['t'] = 4
    assert not inventory(g, g)[1]['observed_consecutive_scope']


def test_third_branch_fails_instead_of_silently_reducing_degree():
    g = graph()
    g['edges'].append(dict(source_id=1, target_id=4))
    with pytest.raises(ValueError, match='degree'):
        inventory(g, g)
