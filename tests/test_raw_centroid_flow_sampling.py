import numpy as np
import pytest

from research.raw_centroid_flow_sampling import sample_raw_centroid_flow


def affine_field():
    z, y, x = np.meshgrid(np.arange(4), np.arange(3), np.arange(3), indexing='ij')
    return np.stack([z + 2*y + 3*x, -z + y, 2*x - y]).astype(np.float32)


def test_exact_fractional_physical_vectors_and_nonmutation():
    points = np.array([[1.25, 3., 6.], [0., 0., 0.], [3., 8., 8.]])
    before = points.copy()
    out, receipt = sample_raw_centroid_flow(affine_field(), points, (4, 12, 12))
    z, y, x = (points / [1, 4, 4]).T
    np.testing.assert_allclose(out, np.stack([z+2*y+3*x, -z+y, 2*x-y], axis=1))
    np.testing.assert_array_equal(points, before)
    assert receipt['trailing_border_extended_nodes'] == 0
    assert receipt['coordinates_modified'] is False


def test_trailing_strip_is_explicit_and_not_coordinate_clipping():
    points = np.array([[2., 11., 10.], [3., 8.25, 8.]])
    before = points.copy()
    out, receipt = sample_raw_centroid_flow(affine_field(), points, (4, 12, 12))
    expected, _ = sample_raw_centroid_flow(affine_field(), np.array([[2,8,8],[3,8,8]]), (4,12,12))
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(points, before)
    assert receipt['trailing_border_extended_nodes'] == 2
    assert receipt['nodes_deleted'] is False


def test_empty_nodes_and_duplicate_identity_are_preserved():
    out, receipt = sample_raw_centroid_flow(affine_field(), np.empty((0,3)), (4,12,12))
    assert out.shape == (0,3) and receipt['sampled_nodes'] == 0
    out, receipt = sample_raw_centroid_flow(affine_field(), [[1,2,3],[1,2,3]], (4,12,12))
    np.testing.assert_array_equal(out[0], out[1])
    assert receipt['sampled_nodes'] == 2


@pytest.mark.parametrize('points', [[[4,0,0]], [[0,12,0]], [[-0.1,0,0]], [[np.nan,0,0]], [[1,2]]])
def test_invalid_centroids_rejected(points):
    with pytest.raises(ValueError, match='centroids'):
        sample_raw_centroid_flow(affine_field(), points, (4,12,12))


def test_wrong_grid_and_nonfinite_field_rejected():
    with pytest.raises(ValueError, match='strided grid'):
        sample_raw_centroid_flow(affine_field()[:,:,:,:2], [[1,2,3]], (4,12,12))
    field = affine_field(); field[0,0,0,0] = np.inf
    with pytest.raises(ValueError, match='strided grid'):
        sample_raw_centroid_flow(field, [[1,2,3]], (4,12,12))


@pytest.mark.parametrize('shape,stride', [((4,12,12),(1,0,4)), ((4,12.5,12),(1,4,4)), ((4,12,12),(1,4.5,4))])
def test_shape_contract(shape, stride):
    with pytest.raises(ValueError):
        sample_raw_centroid_flow(affine_field(), [[1,2,3]], shape, stride)
