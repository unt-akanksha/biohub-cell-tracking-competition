import numpy as np
import pytest
from research.training_match_diagnostic import compare_matches


def test_nearest_only_rule_can_drop_second_valid_parent_label():
    row=compare_matches([[0,0,0],[1,0,0]],[[0,0,0],[3,0,0]],radius_um=3)
    assert row['greedy_matched']==1 and row['exact_matched']==2
    assert row['additional_matched']==1


def test_simple_unique_and_outside_radius():
    row=compare_matches([[0,0,0],[20,0,0]],[[0,0,0],[100,0,0]])
    assert row['greedy_matched']==row['exact_matched']==1
    assert row['changed_assignments']==0


def test_empty_and_invalid_coordinates():
    assert compare_matches(np.empty((0,3)),[[0,0,0]])['exact_matched']==0
    assert compare_matches([[0,0,0]],np.empty((0,3)))['exact_matched']==0
    with pytest.raises(ValueError): compare_matches([[np.nan,0,0]],[[0,0,0]])
    with pytest.raises(ValueError): compare_matches([[0,0]],[[0,0,0]])


def test_dense_assignment_never_loses_greedy_cardinality():
    generator=np.random.default_rng(214)
    for _ in range(20):
        row=compare_matches(generator.normal(size=(30,3)),generator.normal(size=(12,3)),radius_um=.9)
        assert row['exact_matched']>=row['greedy_matched']
