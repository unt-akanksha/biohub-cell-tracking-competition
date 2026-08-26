from __future__ import annotations

import numpy as np
import pytest

from research.spatialdino_association.appearance import (
    edge_cosine_scores,
    parent_choice_margins,
    sample_patch_embeddings,
)


def test_patch_centers_sample_exact_spatial_tokens() -> None:
    grid = np.zeros((3, 2, 2, 2), dtype=np.float32)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                grid[:, z, y, x] = (z, y, x)
    coords = np.asarray([[3.5, 3.5, 3.5], [11.5, 11.5, 11.5]], np.float32)

    sampled = sample_patch_embeddings(
        grid,
        coords,
        input_shape_zyx=(16, 16, 16),
        patch_channels=3,
        normalize=False,
    )

    np.testing.assert_allclose(sampled, [[0, 0, 0], [1, 1, 1]], atol=1e-6)


def test_midpoint_trilinearly_interpolates_tokens() -> None:
    grid = np.zeros((1, 2, 2, 2), dtype=np.float32)
    grid[0, 1, 1, 1] = 8.0

    sampled = sample_patch_embeddings(
        grid,
        np.asarray([[7.5, 7.5, 7.5]], np.float32),
        input_shape_zyx=(16, 16, 16),
        patch_channels=1,
        normalize=False,
    )

    np.testing.assert_allclose(sampled, [[1.0]], atol=1e-6)


def test_sampled_embeddings_are_normalized_by_default() -> None:
    grid = np.zeros((2, 1, 1, 1), dtype=np.float32)
    grid[:, 0, 0, 0] = (3.0, 4.0)

    sampled = sample_patch_embeddings(
        grid,
        np.asarray([[3.5, 3.5, 3.5]], np.float32),
        input_shape_zyx=(8, 8, 8),
        patch_channels=2,
    )

    np.testing.assert_allclose(sampled, [[0.6, 0.8]], atol=1e-6)


def test_cosine_and_parent_margins_are_candidate_local() -> None:
    embeddings = np.asarray([[1, 0], [0.8, 0.2], [0, 1], [1, 0]], np.float32)
    edges = np.asarray([[0, 3], [1, 3], [2, 3]], np.int64)

    scores = edge_cosine_scores(embeddings, edges)
    margins = parent_choice_margins(scores, edges)

    assert scores[0] == pytest.approx(1.0)
    assert margins[0] > 0
    assert margins[1] < 0
    assert margins[2] < margins[1]


def test_sampling_rejects_mismatched_grid_and_out_of_bounds_nodes() -> None:
    with pytest.raises(ValueError, match="does not match"):
        sample_patch_embeddings(
            np.zeros((3, 2, 2, 2), np.float32),
            np.zeros((1, 3), np.float32),
            input_shape_zyx=(24, 16, 16),
            patch_channels=3,
        )
    with pytest.raises(ValueError, match="outside"):
        sample_patch_embeddings(
            np.zeros((3, 2, 2, 2), np.float32),
            np.asarray([[16, 0, 0]], np.float32),
            input_shape_zyx=(16, 16, 16),
            patch_channels=3,
        )


def test_edge_helpers_reject_nonfinite_inputs() -> None:
    edges = np.asarray([[0, 1]], np.int64)
    with pytest.raises(ValueError, match="must be finite"):
        edge_cosine_scores(np.asarray([[1, 0], [np.nan, 1]], np.float32), edges)
    with pytest.raises(ValueError, match="must be finite"):
        parent_choice_margins(np.asarray([np.inf], np.float32), edges)
