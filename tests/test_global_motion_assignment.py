from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from global_motion_assignment import assign,link


def test_global_assignment_resolves_competing_greedy_choices():
    assert assign([[1.,1.1],[1.2,4.]],4.5)==[(0,1),(1,0)]


def test_each_child_has_null_option_and_equal_null_cost_is_not_forced():
    assert assign([[0.,4.5,8.]],4.5)==[(0,0)]
    assert assign(np.empty((0,2)),4.5)==[]
    assert assign(np.empty((2,0)),4.5)==[]


def test_no_forks_gaps_or_coordinate_edits():
    coords=np.array([[0,0,0,0],[1,0,0,0],[1,0,0,1],[3,0,0,0]],float)
    original=coords.copy()
    assert link(coords)==[(0,1)]
    np.testing.assert_array_equal(coords,original)


@pytest.mark.parametrize('costs',[[[float('nan')]], [[-1.]], [[float('inf')]], [1.,2.]])
def test_bad_costs_rejected(costs):
    with pytest.raises(ValueError): assign(costs,4.5)


def test_memory_guard_does_not_truncate_or_drop_nodes():
    with pytest.raises(ValueError,match='guard'): assign(np.zeros((2049,1)),4.5)
