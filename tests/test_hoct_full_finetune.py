from __future__ import annotations

import numpy as np
import pytest
import torch

from research.hoct_graph.biohub_adapter import HOCTTile, build_window
from research.hoct_graph.train_dual_fold_full import (
    aggregate_metrics,
    core_parent_groups,
    parental_training_loss,
    validate_frame_candidate_recall,
    validate_tile_candidate_recall,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo


def fixture_tile() -> HOCTTile:
    node_ids = np.arange(6, dtype=np.int64)
    times = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int32)
    coords = np.asarray(
        [[2, 2, 2], [8, 8, 8], [3, 2, 2], [4, 2, 2], [9, 8, 8], [15, 15, 15]],
        dtype=np.float32,
    )
    truth = {(0, 2), (0, 3), (1, 4)}
    window = build_window(
        node_ids,
        times,
        coords,
        start=0,
        window_size=2,
        neighbors=4,
        max_distance=80,
        true_edges=truth,
    )
    return HOCTTile(window, np.ones(len(window.edge_indices), dtype=bool))


def test_parental_loss_trains_one_parent_per_target_and_preserves_division() -> None:
    tile = fixture_tile()
    groups = core_parent_groups(tile)
    logits = torch.zeros(len(tile.window.edge_indices), requires_grad=True)
    orphan = torch.zeros(len(tile.window.node_ids), requires_grad=True)

    loss, stats = parental_training_loss(logits, orphan, tile)
    loss.backward()

    assert len(groups) == 3
    assert stats["positive_edges"] == 3
    assert stats["supervised_targets"] == 3
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.count_nonzero(logits.grad) > 0
    assert orphan.grad is not None and torch.count_nonzero(orphan.grad) > 0


def test_candidate_recall_fails_when_a_true_core_edge_is_missing() -> None:
    tile = fixture_tile()
    truth = {(0, 2), (0, 3), (1, 4)}
    validate_tile_candidate_recall(tile, truth)
    missing_source, missing_target = next(iter(truth))
    missing_index = next(
        index
        for index, (source, target) in enumerate(tile.window.edge_indices.tolist())
        if int(tile.window.node_ids[source]) == missing_source
        and int(tile.window.node_ids[target]) == missing_target
    )
    broken_mask = tile.core_edge_mask.copy()
    broken_mask[missing_index] = False
    broken = HOCTTile(tile.window, broken_mask)
    with pytest.raises(RuntimeError, match="omitted"):
        validate_tile_candidate_recall(broken, truth)

    video = GraphVideo(
        "fixture",
        tile.window.node_ids,
        tile.window.times,
        tile.window.node_positions / np.asarray([4.0, 1.0, 1.0]),
        np.asarray(sorted(truth), dtype=np.int64),
    )
    validate_frame_candidate_recall([tile], video, 0, truth)
    with pytest.raises(RuntimeError, match="frame edges"):
        validate_frame_candidate_recall([broken], video, 0, truth)


def test_hoct_metric_aggregate_weights_edges_and_divisions_separately() -> None:
    result = aggregate_metrics(
        [
            {"top1": 1.0, "mrr": 1.0, "division_both": 0.0, "rows": 3, "division_rows": 0},
            {"top1": 0.0, "mrr": 0.5, "division_both": 1.0, "rows": 1, "division_rows": 1},
        ]
    )
    assert result["top1"] == 0.75
    assert result["mrr"] == 0.875
    assert result["division_both"] == 1.0
