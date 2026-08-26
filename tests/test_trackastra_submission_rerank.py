from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from research.trackastra_graph.rerank_submission import (
    load_acceptance_evidence,
    read_submission,
    transfer_raw_edge_probabilities,
    validate_edges,
    write_submission,
)
from research.trackastra_graph.train_biohub_graph_transformer import (
    GraphVideo,
    hybrid_link_movie,
)
from research.trackastra_graph.hybrid_linker import HybridLinkConfig


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


def test_raw_edge_probability_transfer_preserves_postprocess_fallbacks() -> None:
    final = GraphVideo(
        "fixture",
        node_ids=np.array([1, 2, 3, 4]),
        times=np.array([0, 1, 1, 2]),
        coords_voxel=np.zeros((4, 3)),
        edges=np.array([[1, 2], [1, 3], [2, 4]]),
    )
    raw = GraphVideo(
        "fixture",
        node_ids=np.array([1, 2, 3, 4]),
        times=np.array([0, 1, 1, 2]),
        coords_voxel=np.zeros((4, 3)),
        edges=np.array([[1, 2], [1, 3]]),
        edge_probabilities=np.array([0.99, 0.62]),
    )

    report = transfer_raw_edge_probabilities(
        {"fixture": final}, {"fixture": raw}, fallback_probability=0.80
    )

    np.testing.assert_allclose(final.edge_probabilities, [0.99, 0.62, 0.80])
    assert report["fixture"]["transferred_edge_probabilities"] == 2
    assert report["fixture"]["fallback_edges"] == 1


def test_raw_confidence_hybrid_locks_only_the_high_confidence_base_edge() -> None:
    video = GraphVideo(
        "fixture",
        node_ids=np.array([1, 2, 3, 4]),
        times=np.array([0, 0, 1, 1]),
        coords_voxel=np.zeros((4, 3)),
        edges=np.array([[1, 3], [2, 4]]),
        edge_probabilities=np.array([0.99, 0.55]),
    )
    pair_scores = {
        0: (
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([[0.10, 0.95], [0.92, 0.10]], dtype=np.float32),
        )
    }
    config = HybridLinkConfig(
        edge_threshold=0.20,
        base_lock_probability=0.96,
        base_keep_probability=0.75,
        base_bonus=0.0,
        division_threshold=1.1,
        division_ratio=1.0,
        base_division_keep_probability=1.1,
    )

    edges = hybrid_link_movie(
        video,
        pair_scores,
        config,
        use_stored_edge_probabilities=True,
    )

    assert (1, 3) in edges
    assert (2, 4) not in edges


def test_external_acceptance_terminal_is_bound_to_model_hash(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"independent-model")
    import hashlib
    import json

    terminal_path = tmp_path / "acceptance_terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "association_acceptance_passed": True,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "complete_movie_selected": {"method": "raw_confidence_hybrid"},
            }
        ),
        encoding="utf-8",
    )

    terminal = load_acceptance_evidence(terminal_path, model_path)
    assert terminal["complete_movie_selected"]["method"] == "raw_confidence_hybrid"

    model_path.write_bytes(b"mutated-model")
    with pytest.raises(RuntimeError, match="does not match"):
        load_acceptance_evidence(terminal_path, model_path)
