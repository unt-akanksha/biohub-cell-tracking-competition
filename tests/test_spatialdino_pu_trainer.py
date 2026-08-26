from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from research.spatialdino_detection.data import VALIDATION_STEMS, select_frame_pairs
from research.spatialdino_detection.train_pu_detector import (
    annotations_to_isotropic_grid,
    ema_decay_for_step,
    normalize_spatialdino_frame,
    update_ema,
)


def test_frame_normalization_is_bounded_and_deterministic() -> None:
    values = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    first = normalize_spatialdino_frame(values)
    second = normalize_spatialdino_frame(values.copy())
    np.testing.assert_array_equal(first, second)
    assert float(first.min()) == 0.0
    assert float(first.max()) == 1.0


def test_frame_normalization_rejects_constant_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="intensity range"):
        normalize_spatialdino_frame(np.zeros((64, 64, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        normalize_spatialdino_frame(np.zeros((32, 64, 64), dtype=np.float32))


def test_annotations_map_to_isotropic_downsampled_grid() -> None:
    points = np.asarray([[10, 80, 120], [70, 80, 120]], dtype=np.float32)
    output = annotations_to_isotropic_grid(points)
    np.testing.assert_allclose(output, [[10, 20, 30]])


def test_ema_updates_floating_state_and_rejects_bad_decay() -> None:
    student = torch.nn.Linear(2, 1)
    ema = torch.nn.Linear(2, 1)
    with torch.no_grad():
        student.weight.fill_(2.0)
        student.bias.fill_(3.0)
        ema.weight.zero_()
        ema.bias.zero_()
    update_ema(ema, student, decay=0.5)
    torch.testing.assert_close(ema.weight, torch.ones_like(ema.weight))
    torch.testing.assert_close(ema.bias, torch.full_like(ema.bias, 1.5))
    with pytest.raises(ValueError, match="decay"):
        update_ema(ema, student, decay=1.0)


def test_frame_pair_inventory_excludes_every_validation_movie() -> None:
    excluded = next(iter(VALIDATION_STEMS))
    selected = select_frame_pairs(
        {"44b6_train": 5, "6bba_train": 7, excluded: 9},
        pairs_per_movie=1,
        seed=11,
    )
    assert {stem for stem, _ in selected} == {"44b6_train", "6bba_train"}
    assert len(selected) == 2
    by_stem = dict(selected)
    for stem, frame_count in (("44b6_train", 5), ("6bba_train", 7)):
        digest = hashlib.sha256(f"11:{stem}".encode("utf-8")).digest()
        assert by_stem[stem] == int.from_bytes(digest[:8], "big") % (frame_count - 1)


def test_ema_decay_warms_up_without_exceeding_maximum() -> None:
    assert ema_decay_for_step(0.995, 1) == pytest.approx(2 / 11)
    assert ema_decay_for_step(0.995, 768) == pytest.approx(769 / 778)
    assert ema_decay_for_step(0.90, 10_000) == 0.90
    with pytest.raises(ValueError, match="positive"):
        ema_decay_for_step(0.995, 0)
