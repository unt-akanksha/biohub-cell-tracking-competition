from __future__ import annotations

import numpy as np
import pytest

from research.spotiflow_biohub.train_pu_detector import (
    VALIDATION_STEMS,
    annotations_to_output_grid,
    select_frame_pairs,
)


def test_frame_pair_selection_excludes_all_validation_movies() -> None:
    counts = {
        "44b6_train_a": 10,
        "6bba_train_b": 8,
        next(iter(VALIDATION_STEMS)): 20,
    }
    first = select_frame_pairs(counts, pairs_per_movie=3, seed=17)
    second = select_frame_pairs(counts, pairs_per_movie=3, seed=17)
    assert first == second
    assert len(first) == 6
    assert {stem for stem, _ in first} == {"44b6_train_a", "6bba_train_b"}
    assert all(0 <= frame < counts[stem] - 1 for stem, frame in first)


def test_annotations_map_from_original_voxels_to_crop_output() -> None:
    points = np.asarray(
        [
            [12, 80, 120],  # in crop -> [2, 10, 15]
            [9, 80, 120],  # before crop
            [50, 80, 120],  # after crop
        ],
        dtype=np.float32,
    )
    output = annotations_to_output_grid(
        points,
        z_start=10,
        input_shape=(32, 64, 64),
        output_shape=(16, 32, 32),
    )
    np.testing.assert_allclose(output, [[1, 10, 15]])


def test_empty_or_invalid_pair_inventory_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="no non-validation"):
        select_frame_pairs({next(iter(VALIDATION_STEMS)): 5}, pairs_per_movie=1, seed=1)
    with pytest.raises(ValueError, match="fewer than two"):
        select_frame_pairs({"44b6_train": 1}, pairs_per_movie=1, seed=1)
