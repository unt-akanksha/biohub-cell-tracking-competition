from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research.trackastra_graph.train_biohub_graph_transformer import (
    WindowSample,
    build_synthetic_video,
)
from research.trackastra_graph.train_dual_fold_synthetic import (
    OPENED_ACCEPTANCE_STEMS,
    augment_global_motion,
    ranking_metrics,
    select_real_paths,
    synthetic_video_native_geometry,
    validation_schedule,
)


def test_synthetic_geometry_is_restored_to_native_trackastra_frame(tmp_path: Path) -> None:
    path = tmp_path / "sequence.npz"
    nodes = np.asarray(
        [[0, 3, 120, 200, 7], [1, 4, 124, 204, 7]], dtype=np.float32
    )
    np.savez(
        path,
        nodes=nodes,
        edges=np.asarray([[0, 1]], dtype=np.int32),
        divisions=np.asarray([], dtype=np.int32),
        voxel_um_pooled=np.asarray([1.625, 1.625, 1.625], dtype=np.float32),
    )

    video = synthetic_video_native_geometry(path)

    np.testing.assert_allclose(video.coords_voxel[0], [3, 120, 200])
    np.testing.assert_allclose(video.scaled_coords[0], [12, 120, 200])


def test_real_fold_selection_excludes_every_opened_acceptance_stem(tmp_path: Path) -> None:
    for stem in ["44b6_a", "44b6_b", *OPENED_ACCEPTANCE_STEMS]:
        (tmp_path / f"{stem}.geff").mkdir()

    selected = select_real_paths(tmp_path, prefix="44b6")

    assert {path.stem for path in selected} == {"44b6_a", "44b6_b"}
    assert not ({path.stem for path in selected} & OPENED_ACCEPTANCE_STEMS)


def test_global_motion_keeps_targets_and_moves_each_frame_rigidly() -> None:
    sample = WindowSample(
        coords=np.asarray(
            [[0, 0, 0, 0], [0, 1, 2, 3], [1, 2, 3, 4], [1, 4, 6, 8]],
            dtype=np.float32,
        ),
        features=np.zeros((4, 12), dtype=np.float32),
        target=np.eye(4, dtype=np.float32),
        valid_mask=np.ones((4, 4), dtype=bool),
        division_target=np.zeros((4, 4), dtype=np.float32),
        positive_edges=4,
        origin_ids=np.arange(4),
    )
    moved = augment_global_motion(
        sample,
        np.random.default_rng(7),
        probability=1.0,
        step_sigma=2.0,
        jump_probability=1.0,
        jump_sigma=3.0,
    )

    np.testing.assert_array_equal(moved.target, sample.target)
    np.testing.assert_allclose(
        moved.coords[1, 1:] - moved.coords[0, 1:],
        sample.coords[1, 1:] - sample.coords[0, 1:],
    )
    assert not np.allclose(moved.coords[:, 1:], sample.coords[:, 1:])


def test_ranking_metric_rewards_true_edge_ordering() -> None:
    video = build_synthetic_video()
    node_count = len(video.node_ids)
    target = np.zeros((node_count, node_count), dtype=np.float32)
    valid = np.zeros_like(target, dtype=bool)
    for source, destination in video.edges:
        source_index = int(np.flatnonzero(video.node_ids == source)[0])
        destination_index = int(np.flatnonzero(video.node_ids == destination)[0])
        target[source_index, destination_index] = 1
        valid[source_index] = True
    coords = np.concatenate(
        [video.times[:, None].astype(np.float32), video.scaled_coords], axis=1
    )
    sample = WindowSample(
        coords=coords,
        features=np.zeros((node_count, 12), dtype=np.float32),
        target=target,
        valid_mask=valid,
        division_target=np.zeros_like(target),
        positive_edges=int(target.sum()),
        origin_ids=video.node_ids,
    )

    class FixedModel(torch.nn.Module):
        def __init__(self, correct: bool) -> None:
            super().__init__()
            self.correct = correct

        def forward(self, coords, features):
            scores = torch.zeros((1, node_count, node_count), device=coords.device)
            truth = torch.from_numpy(target).to(coords.device)
            return scores + (truth * (4 if self.correct else -4))

        def normalize_output(self, logits, timepoints, coords):
            return torch.sigmoid(logits)

    good = ranking_metrics(FixedModel(True), [sample], torch.device("cpu"))
    bad = ranking_metrics(FixedModel(False), [sample], torch.device("cpu"))

    assert good["composite"] > bad["composite"]
    assert good["edge_top1_accuracy"] == 1.0


def test_validation_schedule_includes_early_and_terminal_checkpoints() -> None:
    assert validation_schedule(500, 75000)
    assert validation_schedule(1500, 75000)
    assert validation_schedule(5000, 75000)
    assert validation_schedule(75000, 75000)
    assert not validation_schedule(501, 75000)
