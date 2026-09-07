from __future__ import annotations

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from research.peak_rank_detection import association_bridge as bridge


def test_peak_rank_cache_covers_movie_and_uses_metadata_density(monkeypatch) -> None:
    rows = [
        SimpleNamespace(
            frame=0,
            points_input=np.asarray([[1.25, 2.0, 3.5], [4.0, 5.0, 6.0]]),
            probabilities=np.asarray([0.9, 0.1]),
        ),
        SimpleNamespace(
            frame=1,
            points_input=np.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
            probabilities=np.asarray([0.8, 0.2]),
        ),
    ]
    monkeypatch.setattr(bridge, "range_frame_count", lambda _path: range(2))
    monkeypatch.setattr(
        bridge,
        "predict_frames",
        lambda *_args, **_kwargs: (rows, 2),
    )
    monkeypatch.setattr(bridge, "estimated_count_for_movie", lambda _path: 2.0)

    cache = bridge.predict_movie_detection_cache(
        object(), bridge.Path("movie.zarr"), device="cuda:0", calibration_frames=2
    )

    assert cache.candidate.name == "temporal_peak_rank_v1"
    assert cache.projected_node_count == 2.0
    np.testing.assert_array_equal(cache.association_coords(0), [[0, 1, 2, 4]])
    np.testing.assert_allclose(
        cache.association_coords(0, mode="subvoxel"),
        [[0.0, 1.25, 2.0, 3.5]],
    )
    with pytest.raises(ValueError, match="unsupported association coordinate mode"):
        cache.association_coords(0, mode="unknown")
    np.testing.assert_allclose(
        cache.precise_output_coords(),
        [[0.0, 1.25, 8.0, 14.0], [1.0, 2.0, 12.0, 16.0]],
    )


def test_peak_rank_composition_preserves_linker_edges_and_precision(monkeypatch) -> None:
    rows = [
        SimpleNamespace(
            frame=0,
            points_input=np.asarray([[1.25, 2.0, 3.5]]),
            probabilities=np.asarray([0.9]),
        ),
        SimpleNamespace(
            frame=1,
            points_input=np.asarray([[2.0, 3.0, 4.0]]),
            probabilities=np.asarray([0.8]),
        ),
    ]
    monkeypatch.setattr(bridge, "range_frame_count", lambda _path: range(2))
    monkeypatch.setattr(
        bridge,
        "predict_frames",
        lambda *_args, **_kwargs: (rows, 2),
    )
    monkeypatch.setattr(bridge, "estimated_count_for_movie", lambda _path: 2.0)

    module = ModuleType("official_predict")
    module._detect_cells_pooled = lambda *_args: None

    def predict_video(_model, _path, _device, _cfg, **_kwargs):
        first = module._detect_cells_pooled(None, 0, 0.5, (1, 1, 1))
        second = module._detect_cells_pooled(None, 1, 0.5, (1, 1, 1))
        coordinates = np.concatenate((first, second))
        return coordinates, [(0, 1, 0.7, 4.0)]

    module.predict_video = predict_video
    coordinates, edges = bridge.predict_video_with_peak_rank_detections(
        module,
        object(),
        object(),
        bridge.Path("movie.zarr"),
        "cuda:0",
        object(),
        calibration_frames=2,
    )

    assert edges == [(0, 1, 0.7, 4.0)]
    np.testing.assert_allclose(
        coordinates,
        [[0.0, 1.25, 8.0, 14.0], [1.0, 2.0, 12.0, 16.0]],
    )


def test_subvoxel_association_rounds_feature_sampling_but_not_edge_coordinates() -> None:
    frames = (
        bridge.FrameDetections(
            frame=0,
            points_input=np.asarray([[1.25, 2.0, 3.5]], dtype=np.float32),
            confidence=np.asarray([0.9], dtype=np.float32),
        ),
        bridge.FrameDetections(
            frame=1,
            points_input=np.asarray([[1.75, 2.4, 3.1]], dtype=np.float32),
            confidence=np.asarray([0.8], dtype=np.float32),
        ),
    )
    cache = bridge.ExternalDetectionCache(
        candidate=bridge.PEAK_RANK_CANDIDATE,
        frames=frames,
        threshold=0.5,
        projected_node_count=2.0,
        estimated_node_count=2.0,
    )

    class Linker:
        def __init__(self) -> None:
            self.sampled = []

        def _index_features(self, _maps, coordinates, _mask):
            self.sampled.append(coordinates.clone())
            return coordinates

    primary = Linker()
    module = ModuleType("official_predict")
    module._detect_cells_pooled = lambda *_args: None

    def predict_video(model, _path, _device, _cfg, **_kwargs):
        first = module._detect_cells_pooled(None, 0, 0.5, (1, 1, 1))
        second = module._detect_cells_pooled(None, 1, 0.5, (1, 1, 1))
        first_tensor = torch.from_numpy(first[:, 1:]).unsqueeze(0)
        second_tensor = torch.from_numpy(second[:, 1:]).unsqueeze(0)
        mask = torch.ones((1, 1), dtype=torch.bool)
        model._index_features(None, first_tensor, mask)
        model._index_features(None, second_tensor, mask)
        assert first[0, 1:].tolist() == pytest.approx([1.25, 2.0, 3.5])
        assert second[0, 1:].tolist() == pytest.approx([1.75, 2.4, 3.1])
        return np.concatenate((first, second)), [(0, 1, 0.7, 1.0)]

    module.predict_video = predict_video
    coordinates, _edges = bridge.predict_video_with_external_detections(
        module,
        primary,
        bridge.Path("movie.zarr"),
        "cuda:0",
        object(),
        cache,
        association_coordinate_mode="subvoxel",
    )

    np.testing.assert_array_equal(primary.sampled[0], [[[1.0, 2.0, 4.0]]])
    np.testing.assert_array_equal(primary.sampled[1], [[[2.0, 2.0, 3.0]]])
    np.testing.assert_allclose(
        coordinates,
        [[0.0, 1.25, 8.0, 14.0], [1.0, 1.75, 9.6, 12.4]],
    )
    assert "_index_features" not in vars(primary)
