import torch

from research.peak_rank_detection.model import count_parameters
from research.peak_rank_detection.model_safe_rank import (
    MODEL_FAMILY,
    SafeRankMultiscaleBlobGlobalDetector,
)


def test_safe_rank_model_exposes_detached_evidence() -> None:
    model = SafeRankMultiscaleBlobGlobalDetector(
        widths=(8, 16, 32, 64), depths=(1, 1, 1, 1), global_blocks=1
    )
    frames = torch.rand(1, 3, 16, 16, 16, requires_grad=True)
    output = model(frames)
    assert output["logits"].shape == (1, 1, 16, 16, 16)
    assert output["safe_negative_evidence"].shape == (1, 1, 16, 16, 16)
    assert output["safe_negative_evidence"].requires_grad is False
    output["logits"].mean().backward()
    assert torch.isfinite(frames.grad).all()


def test_safe_rank_model_preserves_large_parameter_contract() -> None:
    model = SafeRankMultiscaleBlobGlobalDetector(
        widths=(128, 256, 512, 1024), depths=(3, 3, 9, 3)
    )
    assert MODEL_FAMILY == "safe_rank_multiscale_blob_global_temporal_peak_rank_v17"
    assert count_parameters(model) == 83_802_246
