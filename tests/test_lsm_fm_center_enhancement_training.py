from __future__ import annotations

import numpy as np
import pytest
import torch

from research.lsm_fm_detection.train_center_enhancement import (
    augment_patches,
    select_annotated_frames,
    split_examples,
)


def test_annotated_frame_selection_is_temporally_spread() -> None:
    annotations = {frame: np.ones((1, 3)) for frame in range(10)}
    assert select_annotated_frames(annotations, maximum_frames=4) == [0, 3, 6, 9]


def test_annotated_frame_selection_drops_empty_frames() -> None:
    annotations = {0: np.empty((0, 3)), 4: np.ones((1, 3))}
    assert select_annotated_frames(annotations, maximum_frames=4) == [4]


def test_split_examples_is_disjoint_and_reproducible() -> None:
    first = split_examples(100, seed=7)
    second = split_examples(100, seed=7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not set(first[0]) & set(first[1])
    assert sorted(np.concatenate(first).tolist()) == list(range(100))


def test_augmentation_flips_patch_and_corresponding_offset(monkeypatch) -> None:
    calls = iter([torch.tensor([True]), torch.tensor([False]), torch.tensor([False])])

    def fake_rand(*args, **kwargs):
        del args, kwargs
        return next(calls).float() * 0.0

    monkeypatch.setattr(torch, "rand", fake_rand)
    patches = torch.arange(2 * 3 * 3 * 3).reshape(1, 2, 3, 3, 3).float()
    offsets = torch.tensor([[1.0, 2.0, 3.0]])
    augmented, adjusted = augment_patches(
        patches, offsets, generator=torch.Generator()
    )
    # fake_rand returns zero, so every axis is flipped by the < 0.5 test.
    assert torch.equal(augmented, torch.flip(patches, dims=(2, 3, 4)))
    torch.testing.assert_close(adjusted, -offsets)


def test_too_few_examples_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="too few"):
        split_examples(7, seed=1)
