from __future__ import annotations

from pathlib import Path

import numpy as np

from research.temporal_localization.score_real_development_probe import (
    cached_interior_frames,
    development_gate,
    frame_match,
    member_device_index,
    summarize_matches,
)


def test_cached_frames_require_complete_temporal_triplets(tmp_path: Path) -> None:
    chunk_root = tmp_path / "movie.zarr" / "0" / "c"
    for frame in (2, 3, 4, 8, 9):
        (chunk_root / str(frame)).mkdir(parents=True)
    assert cached_interior_frames(tmp_path / "movie.zarr") == (3,)


def test_frame_matching_is_one_to_one_and_physical() -> None:
    predicted = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    truth = np.asarray([[0, 0, 0], [0, 40, 0]], dtype=np.float32)
    row = frame_match(predicted, truth)
    assert row["matched"] == 1
    assert row["truth"] == 2
    summary = summarize_matches([row])
    assert summary["annotated_node_recall"] == 0.5


def test_development_gate_requires_gain_without_movie_regression() -> None:
    movies = []
    for index in range(2):
        movies.append(
            {
                "baseline": {"matched_nodes": 10, "mean_matched_distance_um": 2.0},
                "candidate": {
                    "matched_nodes": 11 if index == 0 else 10,
                    "mean_matched_distance_um": 1.5,
                },
                "consensus": {"global_gate_passed": True},
            }
        )
    gate = development_gate(movies)
    assert gate["passed"] is True
    movies[1]["candidate"]["matched_nodes"] = 9
    assert development_gate(movies)["passed"] is False


def test_member_devices_support_one_or_many_gpus() -> None:
    assert [member_device_index(index, 1) for index in range(4)] == [0, 0, 0, 0]
    assert [member_device_index(index, 4) for index in range(4)] == [0, 1, 2, 3]
