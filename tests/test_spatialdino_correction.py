from __future__ import annotations

import numpy as np

from research.spatialdino_association.correction import (
    SwapConfig,
    appearance_pair_swaps,
)


def _crossing_case():
    node_ids = np.asarray([1, 2, 3, 4], np.int64)
    times = np.asarray([0, 0, 1, 1], np.int64)
    coords = np.asarray([[0, 0, 0], [0, 4, 0], [0, 1, 0], [0, 3, 0]], np.float32)
    embeddings = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], np.float32)
    edges = np.asarray([[1, 3], [2, 4]], np.int64)
    return node_ids, times, coords, embeddings, edges


def test_appearance_swap_corrects_crossing_without_changing_degrees() -> None:
    node_ids, times, coords, embeddings, edges = _crossing_case()

    corrected, evidence = appearance_pair_swaps(
        node_ids,
        times,
        coords,
        embeddings,
        edges,
        config=SwapConfig(
            min_appearance_gain=0.5,
            max_pair_distance_um=10,
            max_total_distance_increase_um=10,
            base_lock_probability=0.95,
        ),
        edge_probabilities=np.asarray([0.8, 0.8], np.float32),
        voxel_scale_um=(1, 1, 1),
    )

    assert corrected == [(1, 4), (2, 3)]
    assert evidence["selected_swaps"] == 1
    assert evidence["changed_edges"] == 2


def test_locked_or_geometrically_implausible_edges_are_preserved() -> None:
    node_ids, times, coords, embeddings, edges = _crossing_case()
    locked, locked_evidence = appearance_pair_swaps(
        node_ids,
        times,
        coords,
        embeddings,
        edges,
        config=SwapConfig(min_appearance_gain=0.5, base_lock_probability=0.95),
        edge_probabilities=np.asarray([0.99, 0.8], np.float32),
        voxel_scale_um=(1, 1, 1),
    )
    distant, distant_evidence = appearance_pair_swaps(
        node_ids,
        times,
        coords,
        embeddings,
        edges,
        config=SwapConfig(
            min_appearance_gain=0.5,
            max_pair_distance_um=2,
            base_lock_probability=0.95,
        ),
        edge_probabilities=np.asarray([0.8, 0.8], np.float32),
        voxel_scale_um=(1, 1, 1),
    )

    assert locked == distant == [(1, 3), (2, 4)]
    assert locked_evidence["skipped_locked_edges"] == 1
    assert distant_evidence["candidate_pairs"] == 0


def test_divisions_and_nonconsecutive_edges_are_never_modified() -> None:
    node_ids = np.arange(1, 7, dtype=np.int64)
    times = np.asarray([0, 0, 1, 1, 1, 2], np.int64)
    coords = np.zeros((6, 3), np.float32)
    embeddings = np.eye(6, dtype=np.float32)
    edges = np.asarray([[1, 3], [1, 4], [2, 5], [2, 6]], np.int64)

    corrected, evidence = appearance_pair_swaps(
        node_ids,
        times,
        coords,
        embeddings,
        edges,
        config=SwapConfig(min_appearance_gain=0.1),
    )

    assert corrected == [(1, 3), (1, 4), (2, 5), (2, 6)]
    assert evidence["skipped_degree_edges"] == 3
    assert evidence["skipped_nonconsecutive_edges"] == 1
