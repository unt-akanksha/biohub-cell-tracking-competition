from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from research.lsm_fm_detection.association_bridge import (
    DetectorCandidate,
    ExternalDetectionCache,
    FrameDetections,
    build_detection_cache,
    combine_probabilities,
    predict_video_with_external_detections,
    select_density_threshold,
    substituted_official_detector,
)


def _candidate() -> DetectorCandidate:
    return DetectorCandidate("equal_ensemble_control", 0.5, 0.5)


def _frames() -> tuple[FrameDetections, ...]:
    return (
        FrameDetections(
            0,
            np.asarray([[1.2, 2.25, 3.5], [4.0, 5.0, 6.0]], np.float32),
            np.asarray([0.9, 0.3], np.float32),
        ),
        FrameDetections(
            1,
            np.asarray([[2.1, 3.0, 4.75], [5.0, 6.0, 7.0]], np.float32),
            np.asarray([0.8, 0.2], np.float32),
        ),
    )


def test_candidate_rejects_unusable_blend() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DetectorCandidate("bad", 0.0, 0.0)
    with pytest.raises(ValueError, match="refinement"):
        DetectorCandidate("bad", 1.0, 0.0, "leaderboard_tuned")


def test_combine_probabilities_uses_fixed_normalized_weights() -> None:
    first = np.full((2, 2, 2), 0.2, dtype=np.float32)
    second = np.full((2, 2, 2), 0.8, dtype=np.float32)
    candidate = DetectorCandidate("blend", 1.0, 3.0)
    np.testing.assert_allclose(combine_probabilities(first, second, candidate), 0.65)


def test_density_threshold_is_global_and_strict() -> None:
    threshold, projected = select_density_threshold(
        _frames(), estimated_node_count=2.0, calibration_frames=2
    )
    assert threshold == pytest.approx(0.55)
    assert projected == 2.0


def test_cache_rounds_only_association_coords_and_restores_precision() -> None:
    cache = build_detection_cache(
        _frames(),
        candidate=_candidate(),
        estimated_node_count=2.0,
        calibration_frames=2,
    )
    np.testing.assert_array_equal(cache.association_coords(0), [[0, 1, 2, 4]])
    linked = np.asarray([[0, 1, 8, 16], [1, 2, 12, 20]], dtype=np.int16)
    restored = cache.restore_precise_output(linked)
    np.testing.assert_allclose(
        restored,
        [[0.0, 1.2, 9.0, 14.0], [1.0, 2.1, 12.0, 19.0]],
        atol=1e-6,
    )


def test_cache_rejects_node_identity_change() -> None:
    cache = build_detection_cache(
        _frames(),
        candidate=_candidate(),
        estimated_node_count=2.0,
        calibration_frames=2,
    )
    with pytest.raises(RuntimeError, match="node count"):
        cache.restore_precise_output(np.empty((0, 4), dtype=np.int16))
    with pytest.raises(RuntimeError, match="frame ordering"):
        cache.restore_precise_output(np.asarray([[1, 1, 2, 3], [0, 2, 3, 4]]))


def test_official_detector_substitution_is_scoped_and_restored() -> None:
    module = ModuleType("official_predict")

    def original(*_args):
        return "original"

    module._detect_cells_pooled = original
    cache = build_detection_cache(
        _frames(),
        candidate=_candidate(),
        estimated_node_count=2.0,
        calibration_frames=2,
    )
    with substituted_official_detector(module, cache):
        assert module._detect_cells_pooled(None, 0, 0.9, (1, 1, 1)).tolist() == [
            [0, 1, 2, 4]
        ]
    assert module._detect_cells_pooled is original


def test_external_prediction_restores_precision_without_rewriting_edges() -> None:
    module = ModuleType("official_predict")

    def original(*_args):
        raise AssertionError("original detector should be substituted")

    module._detect_cells_pooled = original

    def predict_video(_model, _path, _device, _cfg, **_kwargs):
        first = module._detect_cells_pooled(None, 0, 0.9, (1, 1, 1))
        second = module._detect_cells_pooled(None, 1, 0.9, (1, 1, 1))
        coords = np.concatenate((first, second), axis=0)
        coords[:, 1:] *= np.asarray([1, 4, 4], dtype=np.int16)
        return coords, [(0, 1, 0.75, 4.0)]

    module.predict_video = predict_video
    cache = build_detection_cache(
        _frames(),
        candidate=_candidate(),
        estimated_node_count=2.0,
        calibration_frames=2,
    )
    coords, edges = predict_video_with_external_detections(
        module, object(), None, object(), object(), cache
    )
    assert edges == [(0, 1, 0.75, 4.0)]
    np.testing.assert_allclose(coords[:, 1:], [[1.2, 9.0, 14.0], [2.1, 12.0, 19.0]])
    assert module._detect_cells_pooled is original


def test_cache_requires_complete_ordered_frames() -> None:
    with pytest.raises(ValueError, match="cover every frame"):
        ExternalDetectionCache(
            candidate=_candidate(),
            frames=(_frames()[1],),
            threshold=0.5,
            projected_node_count=1.0,
            estimated_node_count=1.0,
        )
