import numpy as np
import torch

from research.peak_rank_detection.inference import (
    _invert_offset_field,
    _invert_scalar_field,
    _transform_frames,
    normalize_triplet,
    peaks_from_prediction,
    predict_probability_and_offsets,
    resolve_tta_mode,
)


class CoordinateModel(torch.nn.Module):
    def forward(self, frames: torch.Tensor):
        logits = frames[:, 1:2] * 8.0 - 4.0
        offsets = torch.zeros(len(frames), 3, *frames.shape[-3:], device=frames.device)
        offsets[:, 1] = 0.25
        offsets[:, 2] = -0.25
        return {"logits": logits, "offsets": offsets, "auxiliary_logits": ()}


def test_normalize_triplet_has_stable_range() -> None:
    values = np.arange(3 * 8 * 8 * 8, dtype=np.float32).reshape(3, 8, 8, 8)
    normalized = normalize_triplet(values)
    assert normalized.dtype == np.float32
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_scalar_transforms_round_trip() -> None:
    values = torch.arange(3 * 4 * 5).reshape(1, 3, 4, 5).float()
    for rotation in range(4):
        for flip_x in (False, True):
            for flip_z in (False, True):
                transformed = _transform_frames(
                    values,
                    rotation=rotation,
                    flip_x=flip_x,
                    flip_z=flip_z,
                )
                restored = _invert_scalar_field(
                    transformed,
                    rotation=rotation,
                    flip_x=flip_x,
                    flip_z=flip_z,
                )
                assert torch.equal(restored, values)


def test_offset_inversion_changes_vector_basis() -> None:
    values = torch.zeros(1, 3, 4, 4, 4)
    values[:, 1] = 2.0
    restored = _invert_offset_field(values, rotation=1, flip_x=False)
    assert torch.all(restored[:, 1] == 0)
    assert torch.all(restored[:, 2] == -2)


def test_z_flip_offset_inversion_changes_z_basis() -> None:
    values = torch.zeros(1, 3, 4, 4, 4)
    values[:, 0] = 2.0
    restored = _invert_offset_field(
        values,
        rotation=0,
        flip_x=False,
        flip_z=True,
    )
    assert torch.all(restored[:, 0] == -2)
    assert torch.all(restored[:, 1:] == 0)


def test_d4_probability_is_equivariant_for_coordinate_model() -> None:
    frames = torch.zeros(1, 3, 8, 8, 8)
    frames[:, 1, 3, 2, 5] = 1.0
    probability, offsets = predict_probability_and_offsets(
        CoordinateModel(), frames, d4_tta=True
    )
    assert tuple(torch.nonzero(probability == probability.max())[0].tolist()) == (
        0,
        0,
        3,
        2,
        5,
    )
    assert offsets.shape == (1, 3, 8, 8, 8)


def test_explicit_tta_modes_have_frozen_view_counts() -> None:
    class CountingModel(CoordinateModel):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, frames: torch.Tensor):
            self.calls += 1
            return super().forward(frames)

    frames = torch.zeros(1, 3, 8, 8, 8)
    for mode, expected in (
        ("none", 1),
        ("zflip2", 2),
        ("rot4", 4),
        ("d4", 8),
    ):
        model = CountingModel()
        predict_probability_and_offsets(model, frames, tta_mode=mode)
        assert model.calls == expected
    assert resolve_tta_mode(d4_tta=True) == "d4"
    assert resolve_tta_mode(d4_tta=False) == "none"


def test_peak_extraction_applies_learned_offset() -> None:
    probability = torch.zeros(8, 8, 8)
    probability[3, 4, 5] = 0.9
    offsets = torch.zeros(3, 8, 8, 8)
    offsets[:, 3, 4, 5] = torch.tensor([0.25, -0.5, 0.125])
    points, scores = peaks_from_prediction(probability, offsets)
    np.testing.assert_allclose(points, [[3.25, 3.5, 5.125]])
    np.testing.assert_allclose(scores, [0.9])
