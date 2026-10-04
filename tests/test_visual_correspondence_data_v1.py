import numpy as np
import pytest
from research.visual_correspondence_data_v1 import (
    nearest_candidates, supervised_parent, patches_at, validate_roles, image_proposals,
)


def test_truth_not_inserted_into_candidates():
    coords = np.stack((np.arange(30)*.1, np.zeros(30), np.zeros(30)), 1).astype(np.float32)
    selected = nearest_candidates(coords, np.zeros(3))
    assert selected.tolist() == list(range(16))


def test_ambiguous_missing_parent_is_not_negative_or_null():
    assert supervised_parent(np.array([[3., 0, 0]]), np.zeros(3)) is None


def test_known_parent_and_ambiguous_duplicate():
    target, mask = supervised_parent(np.array([[0., 0, 0], [2., 0, 0], [6., 0, 0]]), np.zeros(3))
    assert target == 0
    assert mask.tolist() == [True, False, True]


def test_empty_and_far_parent_candidates_are_known_null():
    assert supervised_parent(np.empty((0, 3)), np.zeros(3))[0] == -1
    target, mask = supervised_parent(np.array([[8., 0, 0]]), np.zeros(3))
    assert target == -1 and mask.all()


def test_patch_axis_and_two_physical_scales():
    image = np.indices((64, 64, 64))[0].astype(np.float32)
    patch = patches_at(image, np.array([[31., 32., 33.]], np.float32))
    assert patch.shape == (1, 2, 15, 15, 15)
    assert patch[0, 0, 8, 7, 7] == 32
    assert patch[0, 1, 8, 7, 7] == 33


def test_proposals_are_image_only():
    image = np.zeros((64, 64, 64), np.float32); image[30, 31, 32] = 1
    assert [30, 31, 32] in image_proposals(image).tolist()


def test_role_and_exclusion_guards():
    with pytest.raises(ValueError, match='leakage'):
        validate_roles([dict(stem='6bba_abc', role='optimization'), dict(stem='6bba_abc', role='selection')])
    with pytest.raises(ValueError, match='Excluded'):
        validate_roles([dict(stem='44b6_12dfb391', role='optimization')])
    with pytest.raises(ValueError, match='sealed'):
        validate_roles([dict(stem='6bba_abc', role='sealed_audit')])


def test_candidate_radius_is_physical():
    selected = nearest_candidates(np.array([[0, 0, 12.], [0, 0, 13.]]), np.zeros(3))
    assert selected.tolist() == [0]
