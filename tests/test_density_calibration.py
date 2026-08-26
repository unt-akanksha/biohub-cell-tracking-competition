from __future__ import annotations

import json

import numpy as np

from research.density_calibration import (
    read_estimated_node_count,
    select_conservative_threshold,
    uniform_frame_indices,
)


def test_uniform_frame_indices_cover_movie_without_end_bias() -> None:
    assert uniform_frame_indices(100, 4).tolist() == [12, 37, 62, 87]
    assert uniform_frame_indices(3, 20).tolist() == [0, 1, 2]


def test_keeps_base_threshold_when_movie_is_not_overcounted() -> None:
    peaks = [np.array([0.99, 0.98]), np.array([0.995])]
    result = select_conservative_threshold(
        peaks,
        estimated_node_count=200,
        n_frames=100,
        base_threshold=0.97,
    )
    assert result.threshold == 0.97
    assert result.reason == "base_not_overcounted"


def test_raises_threshold_to_reduce_projected_overcount() -> None:
    peaks = [
        np.array([0.999, 0.995, 0.990, 0.985, 0.980]),
        np.array([0.998, 0.994, 0.989, 0.984, 0.979]),
    ]
    result = select_conservative_threshold(
        peaks,
        estimated_node_count=200,
        n_frames=100,
        base_threshold=0.97,
        postprocess_retention=1.0,
        overcount_tolerance=0.0,
    )
    assert result.threshold > 0.97
    assert result.calibrated_projected_count == 200.0
    assert result.base_projected_count == 500.0


def test_threshold_increase_is_capped() -> None:
    peaks = [np.linspace(0.971, 0.999, 100)]
    result = select_conservative_threshold(
        peaks,
        estimated_node_count=10,
        n_frames=100,
        base_threshold=0.97,
        postprocess_retention=1.0,
        overcount_tolerance=0.0,
        max_threshold_increase=0.005,
    )
    assert np.isclose(result.threshold, 0.975)
    assert result.reason == "raised_to_safety_cap"


def test_reads_nested_organizer_count_metadata(tmp_path) -> None:
    root = tmp_path / "sample.zarr"
    root.mkdir()
    (root / "zarr.json").write_text(
        json.dumps({"attributes": {"estimated_number_of_nodes": "1234"}}),
        encoding="utf-8",
    )
    assert read_estimated_node_count(root) == 1234.0
