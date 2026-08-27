from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.trackastra_graph.dual_fold_rerank_submission import (
    embryo_prefix,
    movie_inference_weight,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo


def test_embryo_prefix_routes_only_supported_reciprocal_models() -> None:
    assert embryo_prefix("44b6_example") == "44b6"
    assert embryo_prefix("6bba_example") == "6bba"
    with pytest.raises(ValueError, match="no reciprocal model"):
        embryo_prefix("unknown_example")


def test_movie_weight_tracks_pairwise_association_work() -> None:
    video = GraphVideo(
        stem="44b6_cost",
        node_ids=np.arange(9),
        times=np.asarray([0, 0, 1, 1, 1, 2, 2, 2, 2]),
        coords_voxel=np.zeros((9, 3), dtype=np.float32),
        edges=np.asarray([[0, 2], [1, 3], [2, 5], [3, 6], [4, 7]]),
    )

    assert movie_inference_weight(video) == 2 * 3 + 3 * 4
