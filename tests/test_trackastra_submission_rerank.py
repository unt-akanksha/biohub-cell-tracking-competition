from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from research.trackastra_graph.rerank_submission import (
    read_submission,
    validate_edges,
    write_submission,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo


FIXTURE = Path("tests/fixtures/metric/csv_roundtrip/submission.csv")


def test_submission_roundtrip_preserves_nodes_and_replaces_edges(tmp_path: Path) -> None:
    videos = read_submission(FIXTURE)
    assert set(videos) == {"6bba_fixture"}
    video = videos["6bba_fixture"]
    assert video.node_ids.tolist() == [10, 11]
    assert video.times.tolist() == [0, 1]
    assert video.coords_voxel.tolist() == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]

    output = tmp_path / "submission.csv"
    write_submission(output, videos, {"6bba_fixture": [(10, 11)]})
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [row["row_type"] for row in rows] == ["node", "node", "edge"]
    assert [(row["node_id"], row["t"]) for row in rows[:2]] == [("10", "0"), ("11", "1")]
    assert (rows[2]["source_id"], rows[2]["target_id"]) == ("10", "11")


def test_validate_edges_enforces_lineage_topology() -> None:
    video = GraphVideo(
        "fixture",
        node_ids=np.array([1, 2, 3, 4]),
        times=np.array([0, 1, 1, 2]),
        coords_voxel=np.zeros((4, 3)),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    validate_edges(video, [(1, 2), (1, 3), (2, 4)])
    with pytest.raises(ValueError, match="multiple parents"):
        validate_edges(video, [(1, 2), (2, 4), (3, 4)])
    with pytest.raises(ValueError, match="nonconsecutive"):
        validate_edges(video, [(1, 4)])


def test_graph_video_rejects_edge_probability_length_mismatch() -> None:
    with pytest.raises(ValueError, match="edge probability"):
        GraphVideo(
            "fixture",
            node_ids=np.array([1, 2]),
            times=np.array([0, 1]),
            coords_voxel=np.zeros((2, 3)),
            edges=np.array([[1, 2]]),
            edge_probabilities=np.array([]),
        )
