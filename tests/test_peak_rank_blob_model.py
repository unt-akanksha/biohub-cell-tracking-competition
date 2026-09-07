from __future__ import annotations

import torch

from research.peak_rank_detection import evaluate_peak_rank_detector as evaluation
from research.peak_rank_detection.model import count_parameters
from research.peak_rank_detection.model_blob import (
    MODEL_FAMILY,
    BlobAwareTemporalPeakRankDetector,
    blob_bandpass,
    replicated_average,
)


def test_replicated_average_preserves_constant_borders() -> None:
    values = torch.ones((1, 8, 8, 8))
    for kernel in (3, 9):
        assert torch.equal(replicated_average(values, kernel), values)


def test_bandpass_highlights_compact_bright_blob() -> None:
    values = torch.zeros((1, 17, 17, 17))
    values[0, 8, 8, 8] = 1.0
    response = blob_bandpass(values)
    assert float(response[0, 8, 8, 8]) > 0.0
    assert float(response[0, 0, 0, 0]) == 0.0


def test_blob_model_adds_two_channels_and_preserves_output_contract() -> None:
    model = BlobAwareTemporalPeakRankDetector(
        widths=(8, 16, 32, 64), depths=(1, 1, 1, 1)
    )
    frames = torch.rand((1, 3, 16, 16, 16))
    channels = model.temporal_channels(frames)
    output = model(frames)
    assert channels.shape == (1, 8, 16, 16, 16)
    assert output["logits"].shape == (1, 1, 16, 16, 16)
    assert output["offsets"].shape == (1, 3, 16, 16, 16)
    assert len(output["auxiliary_logits"]) == 2
    assert model.stem[0].in_channels == 8
    assert count_parameters(model) > 0
    assert MODEL_FAMILY == "blob_aware_temporal_peak_rank_v11"


def test_evaluator_reconstructs_blob_family_checkpoint(tmp_path) -> None:
    model = BlobAwareTemporalPeakRankDetector(
        widths=(8, 16, 32, 64), depths=(1, 1, 1, 1)
    )
    checkpoint = tmp_path / "blob.pt"
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
    assert isinstance(loaded, BlobAwareTemporalPeakRankDetector)
    assert count_parameters(loaded) == count_parameters(model)
