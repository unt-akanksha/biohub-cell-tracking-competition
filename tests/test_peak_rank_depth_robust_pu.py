from __future__ import annotations

import numpy as np
import torch

from research.peak_rank_detection import train_depth_robust_pu_detector as variant
from research.peak_rank_detection.train_synthetic_real_detector import TrainingExample


class FixedRng:
    def __init__(self, random_values: list[float], uniform_value: float) -> None:
        self.random_values = iter(random_values)
        self.uniform_value = uniform_value

    def random(self) -> float:
        return next(self.random_values)

    def uniform(self, _low: float, _high: float) -> float:
        return self.uniform_value


def test_depth_attenuation_changes_images_without_moving_points(monkeypatch) -> None:
    example = TrainingExample(
        frames=np.ones((3, 4, 3, 3), dtype=np.float32),
        points=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        division_points=np.empty((0, 3), dtype=np.float32),
        source="test",
        identity="test:t1",
    )
    monkeypatch.setattr(
        variant,
        "_base_augment_example",
        lambda row, _rng: (row.frames.copy(), row.points.copy()),
    )
    frames, points = variant.augment_example(
        example,
        FixedRng([0.0, 1.0], 0.25),
    )
    np.testing.assert_allclose(frames[:, 0], 1.0)
    np.testing.assert_allclose(frames[:, -1], 0.25)
    np.testing.assert_array_equal(points, example.points)


def test_sparse_real_loss_has_no_negative_rank_term() -> None:
    logits = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
    offsets = torch.zeros((1, 3, 8, 8, 8), requires_grad=True)
    prediction = {
        "logits": logits,
        "offsets": offsets,
        "auxiliary_logits": (
            torch.zeros((1, 1, 2, 2, 2), requires_grad=True),
            torch.zeros((1, 1, 4, 4, 4), requires_grad=True),
        ),
    }
    points = torch.tensor([[4.0, 4.0, 4.0]])
    loss, components = variant.training_loss(
        prediction,
        points,
        complete_labels=False,
    )
    loss.backward()
    assert components["rank"] == 0.0
    assert float(logits.grad[0, 0, 0, 0, 0]) == 0.0
    assert float(logits.grad[0, 0, 4, 4, 4]) < 0.0


def test_variant_has_distinct_run_identity() -> None:
    assert variant.RUN_ID.endswith("peak-rank-v2")
    assert variant.MINIMUM_DEPTH_FACTOR == 0.25
