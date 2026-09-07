import numpy as np
import torch

from research.peak_rank_detection.diagnose_local_snr_prior import local_snr_band
from research.peak_rank_detection.model import count_parameters
from research.peak_rank_detection.model_temporal_stable import (
    LOCAL_SNR_BANDS,
    MODEL_FAMILY,
    TemporalMinimumLocalSnrSafeRankDetector,
    temporal_minimum_local_snr_channels,
)


def test_temporal_minimum_local_snr_channels_are_stable_bandpasses() -> None:
    frames = torch.zeros((1, 3, 16, 16, 16))
    frames[:, :, 7:10, 7:10, 7:10] = 1.0
    frames[:, 1, 8, 8, 8] = 4.0
    channels = temporal_minimum_local_snr_channels(frames)
    assert channels.shape == (1, 3, 16, 16, 16)
    assert torch.isfinite(channels).all()
    assert torch.all(channels[0, :, 8, 8, 8] > 0.0)
    frames[:, 1, 8, 8, 8] = 100.0
    torch.testing.assert_close(
        channels, temporal_minimum_local_snr_channels(frames)
    )


def test_temporal_minimum_channels_match_cpu_diagnostic_definition() -> None:
    generator = torch.Generator().manual_seed(17)
    frames = torch.rand((1, 3, 16, 16, 16), generator=generator)
    actual = temporal_minimum_local_snr_channels(frames)[0]
    stable = frames.amin(dim=1)[0].numpy()
    expected = torch.from_numpy(
        np.stack(
            [
                local_snr_band(stable, inner, outer)
                for inner, outer in LOCAL_SNR_BANDS
            ]
        )
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert LOCAL_SNR_BANDS == ((3, 9), (3, 11), (5, 13))


def test_temporal_stable_detector_preserves_safe_rank_contract() -> None:
    model = TemporalMinimumLocalSnrSafeRankDetector(
        widths=(8, 16, 32, 64), depths=(1, 1, 1, 1), global_blocks=1
    )
    frames = torch.rand((1, 3, 16, 16, 16))
    channels = model.temporal_channels(frames)
    assert channels.shape == (1, 15, 16, 16, 16)
    assert model._safe_negative_evidence is not None
    model._safe_negative_evidence = None
    output = model(frames)
    assert output["logits"].shape == (1, 1, 16, 16, 16)
    assert output["safe_negative_evidence"].shape == (1, 1, 16, 16, 16)
    assert count_parameters(model) > 0
    assert MODEL_FAMILY.endswith("peak_rank_v23")
