import pytest
import torch
import torch.nn.functional as F

from research.peak_rank_detection.model import count_parameters
from research.peak_rank_detection.model_multiscale import (
    MODEL_FAMILY,
    MultiscaleBlobGlobalTemporalPeakRankDetector,
    multiscale_blob_channels,
    separable_replicated_average,
)


def direct_replicated_average(values: torch.Tensor, kernel: int) -> torch.Tensor:
    padding = kernel // 2
    return F.avg_pool3d(
        F.pad(
            values[:, None],
            (padding, padding, padding, padding, padding, padding),
            mode="replicate",
        ),
        kernel_size=kernel,
        stride=1,
    )[:, 0]


def test_separable_average_matches_direct_box_filter() -> None:
    values = torch.rand(2, 9, 11, 13)
    expected = direct_replicated_average(values, 5)
    actual = separable_replicated_average(values, 5)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_multiscale_channels_have_complementary_blob_responses() -> None:
    frames = torch.zeros(1, 3, 25, 25, 25)
    frames[:, :, 10:15, 10:15, 10:15] = 1.0
    channels = multiscale_blob_channels(frames)
    assert channels.shape == (1, 6, 25, 25, 25)
    assert torch.isfinite(channels).all()
    assert torch.all(channels[0, :, 12, 12, 12] > 0.0)
    assert not torch.allclose(channels[:, 0], channels[:, 2])


def test_multiscale_global_model_forward_and_gradient() -> None:
    model = MultiscaleBlobGlobalTemporalPeakRankDetector(
        widths=(8, 16, 32, 64), depths=(1, 1, 1, 1), global_blocks=1
    )
    frames = torch.rand(1, 3, 16, 16, 16, requires_grad=True)
    output = model(frames)
    assert output["logits"].shape == (1, 1, 16, 16, 16)
    assert output["offsets"].shape == (1, 3, 16, 16, 16)
    assert len(output["auxiliary_logits"]) == 2
    output["logits"].mean().backward()
    assert frames.grad is not None
    assert torch.isfinite(frames.grad).all()


def test_default_multiscale_global_parameter_contract() -> None:
    model = MultiscaleBlobGlobalTemporalPeakRankDetector(
        widths=(128, 256, 512, 1024), depths=(3, 3, 9, 3)
    )
    assert MODEL_FAMILY == "multiscale_blob_global_context_temporal_peak_rank_v15"
    assert count_parameters(model) == 83_802_246


def test_multiscale_model_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        multiscale_blob_channels(torch.zeros(1, 2, 16, 16, 16))
