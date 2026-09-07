from __future__ import annotations

import pytest
import torch

from research.peak_rank_detection import evaluate_peak_rank_detector as evaluation
from research.peak_rank_detection.model import count_parameters
from research.peak_rank_detection.model_global import (
    MODEL_FAMILY,
    BlobGlobalTemporalPeakRankDetector,
    GlobalContextBlock3D,
)


def test_global_block_preserves_volume_shape_and_backpropagates() -> None:
    block = GlobalContextBlock3D(32, heads=8, expansion=2)
    values = torch.randn((1, 32, 2, 2, 2), requires_grad=True)
    output = block(values)
    assert output.shape == values.shape
    output.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_global_detector_has_expected_large_contract() -> None:
    model = BlobGlobalTemporalPeakRankDetector(
        widths=(128, 256, 512, 1024),
        depths=(3, 3, 9, 3),
    )
    assert count_parameters(model) == 83_788_422
    assert MODEL_FAMILY == "blob_global_context_temporal_peak_rank_v13"
    assert len(model.encoder[-1]) == 3


def test_compact_global_detector_output_contract() -> None:
    model = BlobGlobalTemporalPeakRankDetector(
        widths=(8, 16, 32, 64),
        depths=(1, 1, 1, 1),
    ).eval()
    with torch.inference_mode():
        output = model(torch.rand((1, 3, 16, 16, 16)))
    assert output["logits"].shape == (1, 1, 16, 16, 16)
    assert output["offsets"].shape == (1, 3, 16, 16, 16)
    assert len(output["auxiliary_logits"]) == 2


def test_evaluator_reconstructs_global_family_checkpoint(tmp_path) -> None:
    model = BlobGlobalTemporalPeakRankDetector(
        widths=(8, 16, 32, 64), depths=(1, 1, 1, 1)
    )
    checkpoint = tmp_path / "global.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    loaded = evaluation.load_model(
        checkpoint,
        {
            "model_family": MODEL_FAMILY,
            "widths": [8, 16, 32, 64],
            "depths": [1, 1, 1, 1],
            "parameter_count": count_parameters(model),
        },
        torch.device("cpu"),
    )
    assert isinstance(loaded, BlobGlobalTemporalPeakRankDetector)
    assert count_parameters(loaded) == count_parameters(model)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"channels": 30, "heads": 8},
        {"channels": 32, "heads": 8, "expansion": 0},
    ),
)
def test_global_block_rejects_invalid_contract(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        GlobalContextBlock3D(**kwargs)
