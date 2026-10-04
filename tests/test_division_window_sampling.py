import numpy as np
import pytest
from research.division_window_sampling import division_flags, balanced_weights


def test_flags_use_only_annotated_forks_and_do_not_mutate_targets():
    ordinary = np.eye(3)[None]
    division = np.asarray([[[1, 1, 0], [0, 0, 1], [0, 0, 0]]])
    copy = division.copy()
    np.testing.assert_array_equal(division_flags([ordinary, division, np.zeros((1, 3, 3))]), [False, True, False])
    np.testing.assert_array_equal(division, copy)


def test_rare_case_sampling_has_equal_group_mass_not_equal_window_weight():
    flags = np.asarray([True] + [False]*99)
    weights = balanced_weights(flags)
    assert weights[flags].sum() == pytest.approx(.5)
    assert weights[~flags].sum() == pytest.approx(.5)
    assert weights.sum() == pytest.approx(1.)
    assert weights[0] / weights[1] == pytest.approx(99.)


@pytest.mark.parametrize('target', [np.ones((1, 3, 3)), np.eye(3), np.full((1, 2, 2), .5),
    np.full((1, 2, 2), np.nan), np.asarray([[[1, 0], [1, 0]]])])
def test_invalid_lineage_targets_rejected(target):
    with pytest.raises(ValueError):
        division_flags([target])


@pytest.mark.parametrize('flags', [np.asarray([], bool), np.asarray([True, True]),
    np.asarray([False, False]), np.asarray([0, 1])])
def test_sampler_requires_both_valid_groups(flags):
    with pytest.raises(ValueError):
        balanced_weights(flags)
