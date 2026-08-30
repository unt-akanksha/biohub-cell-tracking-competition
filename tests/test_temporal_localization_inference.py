from __future__ import annotations

import numpy as np
import pytest
import torch

from research.temporal_localization.inference import (
    MovieGraphArrays,
    graph_motion_features,
    predict_member,
)


def graph() -> MovieGraphArrays:
    return MovieGraphArrays(
        node_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        times=np.asarray([0, 1, 1, 2], dtype=np.int64),
        coordinates_zyx_voxel=np.asarray(
            [[4, 4, 4], [4, 5, 4], [7, 7, 7], [4, 6, 4]], dtype=np.float32
        ),
        edges_by_node_id=np.asarray([[10, 11], [11, 13]], dtype=np.int64),
    )


class DummyModel(torch.nn.Module):
    def forward(self, patches, graph_features):
        offsets = graph_features[:, :3]
        log_variance = torch.zeros_like(offsets)
        safe = torch.full((len(offsets),), 0.25, device=offsets.device)
        return offsets, log_variance, safe


def test_graph_motion_features_match_parent_child_masks() -> None:
    features = graph_motion_features(
        graph(),
        np.asarray([1, 2]),
        voxel_size_zyx_um=np.ones(3, dtype=np.float32),
        spatial_shape_zyx=(16, 16, 16),
        timepoints=3,
    )
    assert features.shape == (2, 12)
    assert features[0, 6:9].tolist() == [1.0, 1.0, 0.5]
    assert features[1, 6:9].tolist() == [0.0, 0.0, 0.0]


def test_predict_member_preserves_graph_row_order_for_selected_frames() -> None:
    volumes = np.random.default_rng(3).normal(size=(3, 16, 16, 16)).astype(np.float32)
    rows, offsets, sigma, safe = predict_member(
        DummyModel(),
        graph(),
        lambda frame: volumes[frame],
        (1,),
        device=torch.device("cpu"),
        voxel_size_zyx_um=np.ones(3, dtype=np.float32),
        spatial_shape_zyx=(16, 16, 16),
        timepoints=3,
        batch_size=1,
    )
    assert rows.tolist() == [1, 2]
    assert offsets.shape == (2, 3)
    assert np.allclose(sigma, 1.0)
    assert np.allclose(safe, 0.25)


def test_predict_member_rejects_boundary_frames() -> None:
    with pytest.raises(ValueError, match="interior"):
        predict_member(
            DummyModel(),
            graph(),
            lambda _frame: np.zeros((16, 16, 16), dtype=np.float32),
            (0,),
            device=torch.device("cpu"),
            voxel_size_zyx_um=np.ones(3, dtype=np.float32),
            spatial_shape_zyx=(16, 16, 16),
            timepoints=3,
        )
