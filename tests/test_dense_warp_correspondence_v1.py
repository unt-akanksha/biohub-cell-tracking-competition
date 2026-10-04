import numpy as np
import pytest
from research.dense_warp_correspondence_v1 import forward_points, inverse_points, inverse_grid, known_pairs


def test_exact_inverse_for_nonrigid_warp():
    points = np.random.default_rng(2).uniform(0, 63, (400, 3))
    parameters = [2.3, -1.1, .7, 2., -1.8]
    np.testing.assert_allclose(inverse_points(forward_points(points, parameters), parameters), points, atol=2e-14)


def test_grid_xyz_order_and_translation_sign():
    grid = inverse_grid([1, -2, 3, 0, 0], 8)
    np.testing.assert_allclose(grid[4, 3, 5], 2*np.array([2, 5, 3])/7-1, atol=1e-7)


def test_identity_grid_corners():
    grid = inverse_grid([0]*5, 8)
    np.testing.assert_array_equal(grid[0,0,0], [-1]*3)
    np.testing.assert_array_equal(grid[-1,-1,-1], [1]*3)


def test_permuted_identity_and_boundary_exclusion():
    source = np.array([[10, 10, 10], [62, 62, 62], [20, 20, 20]], np.float32)
    target, labels = known_pairs(source, [3,0,0,0,0], [2,1,0])
    np.testing.assert_array_equal(labels, [2,0])
    np.testing.assert_array_equal(target, [[23,20,20],[13,10,10]])


def test_reject_duplicate_target_ids():
    with pytest.raises(ValueError):
        known_pairs(np.zeros((2,3)), [0]*5, [0,0])
