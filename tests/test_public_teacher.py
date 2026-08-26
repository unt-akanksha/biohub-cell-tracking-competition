from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from research.spotiflow_biohub.public_teacher import (
    PublicTeacherDetector,
    load_public_teacher,
    teacher_probabilities,
)


def test_detection_subset_loads_from_checkpoint_with_transformer_entries(tmp_path) -> None:
    source = PublicTeacherDetector(out_channels=4, layers=(4, 8))
    state = OrderedDict(source.state_dict())
    state["transformer.unused"] = torch.ones(1)
    path = tmp_path / "teacher.pth"
    torch.save(state, path)
    loaded = load_public_teacher(path, out_channels=4, layers=(4, 8))
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_missing_detector_tensor_is_rejected(tmp_path) -> None:
    source = PublicTeacherDetector(out_channels=4, layers=(4, 8))
    state = OrderedDict(source.state_dict())
    state.pop("detect_head.bias")
    path = tmp_path / "teacher.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="detect_head.bias"):
        load_public_teacher(path, out_channels=4, layers=(4, 8))


def test_yx_tta_probabilities_keep_shape_and_range() -> None:
    model = PublicTeacherDetector(out_channels=4, layers=(4, 8)).eval()
    frames = torch.randn(1, 2, 4, 8, 8)
    probabilities = teacher_probabilities(model, frames, yx_tta=True)
    assert probabilities.shape == (1, 2, 4, 8, 8)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
