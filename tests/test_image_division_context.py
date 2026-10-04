import numpy as np
import pytest
from research.image_division_context import image_context


def anchors():
    a = np.zeros((3, 8), dtype=np.float32)
    a[0, 5] = 1
    a[1:, 6] = 1
    a[1:, 0] = .5
    a[1, 3] = -.1
    a[2, 3] = .1
    return a


def gaussian(sigma=1.5, center=(8., 7., 6.)):
    grid = np.indices((17, 17, 17)).astype(float)
    dist = sum((grid[d] - center[d])**2 for d in range(3))
    return np.exp(-dist / (2 * sigma**2))


def test_no_graph_neighbors_or_labels_required_and_flat_has_only_anchors():
    tokens, mask = image_context(np.zeros((3, 17, 17, 17)), anchors())
    assert mask.sum() == 3
    assert np.array_equal(tokens[:3], anchors())
    assert not mask[27:].any()
    with pytest.raises(ValueError, match='exactly three'):
        image_context(np.zeros((3, 17, 17, 17)), np.zeros((43, 8)))


def test_affine_brightness_change_preserves_image_features():
    patch = np.stack([gaussian()] * 3)
    a, ma = image_context(patch, anchors())
    b, mb = image_context(3.5 * patch + 14., anchors())
    assert np.array_equal(ma, mb)
    np.testing.assert_allclose(a, b, atol=2e-6)


def test_shrinking_object_has_smaller_halfmax_component():
    patch = np.stack([gaussian(2.), gaussian(1.5), gaussian(1.)])
    tokens, mask = image_context(patch, anchors())
    assert mask[[3, 11, 19]].all()
    assert tokens[3, 4] > tokens[11, 4] > tokens[19, 4]


def test_reflection_moves_image_peak_and_anchor_geometry_together():
    patch = np.stack([gaussian()] * 3)
    a = anchors()
    original, mask = image_context(patch, a)
    a[:, 3] *= -1
    flipped, flipped_mask = image_context(patch[..., ::-1], a)
    expected = original.copy()
    expected[:, 3] *= -1
    assert np.array_equal(mask, flipped_mask)
    np.testing.assert_allclose(expected, flipped, atol=2e-6)


def test_invalid_inputs_fail_closed():
    patch = np.zeros((3, 17, 17, 17));patch[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        image_context(patch, anchors())
