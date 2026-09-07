import numpy as np
import pytest
import torch

from research.peak_rank_detection.model import (
    TemporalPeakRankDetector,
    count_parameters,
)
from research.peak_rank_detection.objectives import (
    focal_heatmap_loss,
    points_to_gaussian_heatmap,
    sparse_peak_ranking_loss,
    subvoxel_offset_loss,
)


def test_temporal_detector_shapes_and_parameter_scale() -> None:
    model = TemporalPeakRankDetector(widths=(8, 16, 32, 64), depths=(1, 1, 1, 1))
    result = model(torch.randn(2, 3, 16, 16, 16))

    assert result["logits"].shape == (2, 1, 16, 16, 16)
    assert result["offsets"].shape == (2, 3, 16, 16, 16)
    assert len(result["auxiliary_logits"]) == 2
    assert result["auxiliary_logits"][0].shape == (2, 1, 4, 4, 4)
    assert result["auxiliary_logits"][1].shape == (2, 1, 8, 8, 8)
    assert count_parameters(model) > 100_000
    assert torch.max(torch.abs(result["offsets"])) <= 0.5


def test_temporal_channels_encode_ordered_differences() -> None:
    frames = torch.stack(
        (torch.zeros(2, 2, 2), torch.ones(2, 2, 2), torch.full((2, 2, 2), 3.0))
    )[None]
    channels = TemporalPeakRankDetector.temporal_channels(frames)

    assert channels.shape == (1, 6, 2, 2, 2)
    assert torch.all(channels[:, 3] == 1)
    assert torch.all(channels[:, 4] == 2)
    assert torch.all(channels[:, 5] == 3)


def test_gaussian_heatmap_preserves_subvoxel_peak() -> None:
    heatmap = points_to_gaussian_heatmap([[4.2, 5.0, 6.0]], (12, 12, 12))

    assert heatmap.shape == (12, 12, 12)
    assert np.unravel_index(np.argmax(heatmap), heatmap.shape) == (4, 5, 6)
    assert heatmap[4, 5, 6] == 1.0
    assert heatmap[0, 0, 0] == 0.0


def test_peak_ranking_rewards_annotated_local_maximum() -> None:
    points = [torch.tensor([[5.0, 5.0, 5.0]])]
    weak = torch.zeros(1, 1, 12, 12, 12, requires_grad=True)
    strong = weak.detach().clone()
    strong[0, 0, 5, 5, 5] = 5.0
    strong.requires_grad_(True)

    weak_loss = sparse_peak_ranking_loss(weak, points)
    strong_loss = sparse_peak_ranking_loss(strong, points)
    assert strong_loss < weak_loss
    strong_loss.backward()
    assert strong.grad is not None


def test_peak_ranking_does_not_use_other_annotation_as_negative() -> None:
    logits = torch.zeros(1, 1, 12, 12, 12)
    logits[0, 0, 5, 5, 5] = 4.0
    logits[0, 0, 5, 5, 8] = 5.0
    points = [torch.tensor([[5.0, 5.0, 5.0], [5.0, 5.0, 8.0]])]

    loss = sparse_peak_ranking_loss(logits, points, exclusion_radius=2.0)
    assert loss < 0.2


def test_focal_and_offset_losses_are_finite() -> None:
    logits = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    target = torch.from_numpy(
        points_to_gaussian_heatmap([[3.25, 4.0, 5.0]], (8, 8, 8))
    )[None, None]
    offsets = torch.zeros(1, 3, 8, 8, 8, requires_grad=True)
    points = [torch.tensor([[3.25, 4.0, 5.0]])]

    loss = focal_heatmap_loss(logits, target) + subvoxel_offset_loss(offsets, points)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert offsets.grad is not None


def test_invalid_detector_inputs_are_rejected() -> None:
    model = TemporalPeakRankDetector(widths=(8, 16, 32, 64), depths=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 1, 16, 16, 16))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 15, 16, 16))
