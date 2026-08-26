from __future__ import annotations

import numpy as np
import pytest
import torch

from research.spatialdino_detection.inference import (
    peaks_from_probability,
    predict_probability_batch,
    refine_peaks_soft_centroid,
)


class IdentityLogit(torch.nn.Module):
    def forward(self, values):
        return values


def test_soft_centroid_refines_asymmetric_peak() -> None:
    probability = np.zeros((5, 5, 5), dtype=np.float32)
    probability[2, 2, 2] = 1.0
    probability[2, 2, 3] = 0.5
    refined = refine_peaks_soft_centroid(probability, np.asarray([[2, 2, 2]]))
    assert refined.shape == (1, 3)
    np.testing.assert_allclose(refined[0, :2], [2, 2], atol=1e-6)
    assert 2.0 < refined[0, 2] < 2.5


def test_peak_extraction_keeps_probability_at_integer_maximum() -> None:
    probability = np.zeros((5, 5, 5), dtype=np.float32)
    probability[1, 2, 3] = 0.9
    points, scores = peaks_from_probability(probability, threshold=0.5)
    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [1, 2, 3])
    np.testing.assert_allclose(scores, [0.9])


def test_probability_tta_is_flip_equivariant_for_identity_logits() -> None:
    image = torch.randn(2, 1, 8, 8, 8)
    expected = torch.sigmoid(image)
    actual = predict_probability_batch(IdentityLogit(), image, yx_tta=True)
    torch.testing.assert_close(actual, expected)


def test_soft_centroid_rejects_invalid_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        refine_peaks_soft_centroid(np.zeros((3, 3, 3)), np.empty((0, 3)), radius=-1)
