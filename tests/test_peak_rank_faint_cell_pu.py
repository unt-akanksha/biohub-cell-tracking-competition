from __future__ import annotations

import numpy as np
import torch

from research.peak_rank_detection import train_faint_cell_pu_detector as variant


def test_local_fade_is_soft_bounded_and_centered() -> None:
    frame = np.ones((17, 17, 17), dtype=np.float32)
    point = np.asarray([8.0, 8.0, 8.0], dtype=np.float32)
    variant.attenuate_local_sphere(
        frame, point, center_factor=0.1, sigma=2.0
    )

    assert np.isclose(frame[8, 8, 8], 0.1)
    assert 0.1 < frame[8, 8, 10] < 1.0
    assert frame[0, 0, 0] == 1.0
    assert float(frame.min()) >= 0.0
    assert float(frame.max()) <= 1.0


def test_temporal_fade_preserves_shape_points_and_some_context() -> None:
    frames = np.ones((3, 17, 17, 17), dtype=np.float32)
    points = np.asarray([[8.0, 8.0, 8.0]], dtype=np.float32)
    faded = variant.apply_temporal_fading(
        frames, points, np.random.default_rng(1041729), probability=1.0
    )

    assert faded.shape == frames.shape
    assert faded.dtype == np.float32
    assert faded[1, 8, 8, 8] < 0.5
    assert np.count_nonzero(faded[0] < 1.0) == 0 or np.count_nonzero(faded[2] < 1.0) == 0
    np.testing.assert_array_equal(points, [[8.0, 8.0, 8.0]])


def test_sparse_real_loss_has_no_unlabeled_negative_gradient() -> None:
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
    loss, components = variant.training_loss(
        prediction, torch.tensor([[4.0, 4.0, 4.0]]), complete_labels=False
    )
    loss.backward()

    assert components["rank"] == 0.0
    assert float(logits.grad[0, 0, 0, 0, 0]) == 0.0
    assert float(logits.grad[0, 0, 4, 4, 4]) < 0.0


def test_variant_contract_is_distinct_and_generic() -> None:
    assert variant.RUN_ID.endswith("faint-temporal-peak-rank-v4")
    assert variant.MAXIMUM_FADED_POINTS == 24
    assert variant.REAL_FADE_PROBABILITY < variant.SYNTHETIC_FADE_PROBABILITY

