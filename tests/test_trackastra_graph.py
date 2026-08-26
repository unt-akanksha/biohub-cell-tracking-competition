from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.trackastra_graph.train_biohub_graph_transformer import (
    _recursive_source_tiles,
    build_synthetic_video,
    compute_division_confusion,
    compute_edge_confusion,
    link_movie,
    match_nodes_bipartite,
    read_synthetic_graph_video,
    sample_window,
    select_synthetic_sequence_paths,
    video_plain,
)


def _clean_sample_kwargs() -> dict:
    return {
        "window": 4,
        "max_tokens": 128,
        "tile_radius": np.array((96.0, 192.0, 192.0), dtype=np.float32),
        "drop_probability": 0.0,
        "false_positive_probability": 0.0,
        "jitter_sigma": 0.0,
        "prefer_division_probability": 1.0,
        "hard_negative_radius": 64.0,
    }


def test_sample_window_builds_trackastra_compatible_targets() -> None:
    video = build_synthetic_video()
    sample = sample_window([video], np.random.default_rng(17), **_clean_sample_kwargs())

    assert sample.coords.ndim == 2
    assert sample.coords.shape[1] == 4
    assert sample.features.shape == (len(sample.coords), 12)
    assert sample.target.shape == (len(sample.coords), len(sample.coords))
    assert sample.positive_edges > 0
    assert np.all(sample.target.astype(bool) <= sample.valid_mask)
    source_rows, target_cols = np.where(sample.target > 0)
    assert np.all(sample.coords[target_cols, 0] - sample.coords[source_rows, 0] == 1)


def test_recursive_tiles_cover_each_source_and_respect_budget() -> None:
    source = np.stack(
        [np.zeros(37), np.linspace(0, 360, 37), np.linspace(0, 180, 37)], axis=1
    ).astype(np.float32)
    target = np.concatenate([source + 1, source + np.array([0, 4, -3])], axis=0)

    tiles = _recursive_source_tiles(source, target, max_tokens=24, radius=30)
    covered = np.concatenate([source_indices for source_indices, _ in tiles])

    assert sorted(covered.tolist()) == list(range(len(source)))
    assert len(np.unique(covered)) == len(source)
    assert all(len(sources) + len(targets) <= 24 for sources, targets in tiles)


def test_link_movie_enforces_one_parent_and_at_most_two_children() -> None:
    source_ids = np.array([10, 11, 12])
    target_ids = np.array([20, 21, 22, 23])
    scores = np.array(
        [
            [0.90, 0.70, 0.01, 0.01],
            [0.10, 0.20, 0.80, 0.01],
            [0.05, 0.04, 0.03, 0.75],
        ],
        dtype=np.float32,
    )

    edges = link_movie(
        {0: (source_ids, target_ids, scores)},
        edge_threshold=0.30,
        division_threshold=0.50,
        division_ratio=0.50,
    )

    assert set(edges) == {(10, 20), (10, 21), (11, 22), (12, 23)}
    assert max(sum(target == candidate for _, target in edges) for candidate in target_ids) == 1
    assert max(sum(source == candidate for source, _ in edges) for candidate in source_ids) == 2


def test_clean_validator_scores_perfect_edges_and_division() -> None:
    video = build_synthetic_video()
    nodes, edges = video_plain(video)
    pred_to_gt, gt_to_pred = match_nodes_bipartite(nodes, nodes)

    assert len(pred_to_gt) == len(nodes)
    assert compute_edge_confusion(edges, edges, pred_to_gt) == (len(edges), 0, 0)
    assert compute_division_confusion(
        nodes, edges, nodes, edges, pred_to_gt, gt_to_pred
    ) == (1, 0, 0)


def test_synthetic_graph_selection_and_geometry_repair(tmp_path: Path) -> None:
    sequences = tmp_path / "sequences"
    sequences.mkdir()
    records = []
    for sequence_index in range(3):
        path = sequences / f"seq_{sequence_index:04d}.npz"
        nodes = np.asarray(
            [
                [0, 3, 120, 200, 7],
                [1, 4, 124, 204, 7],
                [2, 5, 128, 208, 7],
                [3, 6, 132, 212, 7],
            ],
            dtype=np.float32,
        )
        np.savez(
            path,
            nodes=nodes,
            edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            divisions=np.asarray([], dtype=np.int32),
            voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
        )
        records.append(
            {
                "file": f"sequences/{path.name}",
                "T": 4,
                "n_nodes": 4,
                "n_edges": 3,
                "n_divisions": 0,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sequences": records}), encoding="utf-8")

    selected_manifest, paths = select_synthetic_sequence_paths(tmp_path, limit=2)
    assert selected_manifest == manifest
    assert len(paths) == 2
    video = read_synthetic_graph_video(paths[0])
    np.testing.assert_allclose(video.coords_voxel[0], [3, 30, 50])
    np.testing.assert_allclose(video.scaled_coords[0], [12, 30, 50])
    np.testing.assert_array_equal(video.edges, [[0, 1], [1, 2], [2, 3]])
