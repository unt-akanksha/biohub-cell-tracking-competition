from __future__ import annotations

import numpy as np
import torch

from research.hoct_graph.biohub_adapter import (
    BIOHUB_MODEL_SPATIAL_SCALE,
    OFFICIAL_FEATURE_MEAN,
    OFFICIAL_FEATURE_STD,
    build_window,
    candidate_edges,
    iter_pair_tiles,
    pair_score_matrices,
    parental_softmax,
    point_node_features,
    predict_window,
)


def test_point_features_keep_unavailable_morphology_neutral() -> None:
    times = np.asarray([0, 1])
    coords = np.asarray([[2, 10, 20], [3, 11, 21]], dtype=np.float32)
    features, positions = point_node_features(times, coords)

    np.testing.assert_allclose(positions, coords * BIOHUB_MODEL_SPATIAL_SCALE)
    np.testing.assert_allclose(features[:, 4:], 0.0)
    expected_first = (
        np.asarray([0, 8, 10, 20], dtype=np.float32) - OFFICIAL_FEATURE_MEAN[:4]
    ) / OFFICIAL_FEATURE_STD[:4]
    np.testing.assert_allclose(features[0, :4], expected_first)


def test_bidirectional_candidates_cover_sources_and_targets() -> None:
    times = np.asarray([0, 0, 1, 1])
    positions = np.asarray(
        [[0, 0, 0], [0, 20, 0], [0, 1, 0], [0, 19, 0]], dtype=np.float32
    )
    edges, delta = candidate_edges(
        times, positions, neighbors=1, max_distance=5.0, bidirectional=True
    )

    assert set(map(tuple, edges.tolist())) == {(0, 2), (1, 3)}
    np.testing.assert_array_equal(delta, 1.0)


def test_build_window_labels_candidate_edges_by_stable_node_id() -> None:
    window = build_window(
        node_ids=np.asarray([10, 11, 20, 21]),
        times=np.asarray([0, 0, 1, 1]),
        coords_voxel=np.asarray(
            [[0, 0, 0], [0, 20, 0], [0, 1, 0], [0, 19, 0]], dtype=np.float32
        ),
        start=0,
        window_size=2,
        neighbors=1,
        max_distance=5.0,
        true_edges={(10, 20)},
    )

    assert window.edge_labels is not None
    assert window.edge_labels.sum() == 1
    positive = window.edge_indices[np.flatnonzero(window.edge_labels)[0]]
    assert tuple(window.node_ids[positive].tolist()) == (10, 20)


def test_parental_softmax_includes_no_parent_class() -> None:
    edge_prob, orphan_prob = parental_softmax(
        edge_logits=np.asarray([0.0, np.log(2.0), 1.0]),
        orphan_logits=np.asarray([-4.0, -4.0, 0.0]),
        edge_indices=np.asarray([[0, 2], [1, 2], [0, 1]]),
        edge_delta_t=np.ones(3),
    )

    np.testing.assert_allclose(edge_prob[:2].sum() + orphan_prob[2], 1.0)
    np.testing.assert_allclose(edge_prob[2] + orphan_prob[1], 1.0)
    assert edge_prob[1] > edge_prob[0]


class _DummyHOCT(torch.nn.Module):
    def forward(
        self, node_features, node_pos, edge_pos, edge_indices, node_mask, edge_mask
    ):
        batch, nodes, _ = node_features.shape
        edges = edge_indices.shape[1]
        logits = torch.arange(edges, device=node_features.device).reshape(1, edges, 1)
        edge_features = torch.ones((batch, edges, 4), device=node_features.device)
        orphan = torch.zeros((batch, nodes, 1), device=node_features.device)
        return logits, node_features, edge_features, orphan


def test_predict_window_and_dense_pair_scatter() -> None:
    window = build_window(
        node_ids=np.asarray([10, 11, 20, 21]),
        times=np.asarray([0, 0, 1, 1]),
        coords_voxel=np.asarray(
            [[0, 0, 0], [0, 20, 0], [0, 1, 0], [0, 19, 0]], dtype=np.float32
        ),
        start=0,
        window_size=2,
        neighbors=1,
        max_distance=5.0,
    )
    prediction = predict_window(_DummyHOCT(), window, device="cpu")
    matrices = pair_score_matrices(window, prediction.edge_probabilities)

    source_ids, target_ids, scores = matrices[0]
    np.testing.assert_array_equal(source_ids, [10, 11])
    np.testing.assert_array_equal(target_ids, [20, 21])
    assert scores.shape == (2, 2)
    assert np.count_nonzero(scores) == 2


def test_pair_tiles_assign_each_candidate_target_to_one_core() -> None:
    node_ids = np.arange(8)
    times = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    coords = np.asarray(
        [
            [0, 10, 10],
            [0, 60, 10],
            [0, 110, 10],
            [0, 160, 10],
            [0, 11, 10],
            [0, 61, 10],
            [0, 111, 10],
            [0, 161, 10],
        ],
        dtype=np.float32,
    )
    tiles = iter_pair_tiles(
        node_ids,
        times,
        coords,
        source_t=0,
        core_size=np.asarray([128, 100, 128]),
        context_halo=30,
        max_distance=30,
        neighbors=2,
    )

    owned_targets = []
    for tile in tiles:
        owned_targets.extend(
            tile.window.node_ids[
                tile.window.edge_indices[tile.core_edge_mask, 1]
            ].tolist()
        )
    assert set(owned_targets) == {4, 5, 6, 7}
    assert all(owned_targets.count(target_id) >= 1 for target_id in (4, 5, 6, 7))
