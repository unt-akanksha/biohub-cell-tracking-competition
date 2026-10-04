import numpy as np
import pytest

from research.focus_linker_cache import FocusLinkerCache


def node(i, t, x):
    return dict(node_id=i, t=t, z=2.2, y=8.5, x=x)


def test_preserves_every_point_and_empty_frames():
    cache = FocusLinkerCache.from_nodes(
        [node(9, 2, 12.1), node(3, 0, 8.1), node(1, 0, 8.1)], (4, 20, 32, 32))
    assert cache.node_ids == (1, 3, 9)
    assert cache.manifest()['frame_counts'] == [2, 0, 1, 0]
    assert cache.association_coords(1).shape == (0, 4)
    assert cache.association_coords(0).tolist() == [[0, 2, 2, 2], [0, 2, 2, 2]]
    assert not cache.points.flags.writeable


def test_round_trip_and_reorder_guard():
    cache = FocusLinkerCache.from_nodes([node(1, 0, 8), node(2, 0, 24)], (1, 20, 32, 32))
    linked = cache.association_coords(0).astype(float)
    linked[:, 1:] *= cache.downsample
    np.testing.assert_array_equal(cache.restore_precise_output(linked), cache.points)
    np.testing.assert_array_equal(cache.restore_precise_output(cache.points), cache.points)
    with pytest.raises(ValueError, match='order'):
        cache.restore_precise_output(linked[::-1])


@pytest.mark.parametrize('rows', [
    [node(1, 0, -1)], [node(1, 0, 32)], [node(1, .5, 2)],
    [node(1, 0, np.nan)], [node(1, 0, 2), node(1, 0, 3)],
])
def test_invalid_points_fail_without_filtering(rows):
    with pytest.raises(ValueError):
        FocusLinkerCache.from_nodes(rows, (2, 20, 32, 32))


def test_empty_movie_keeps_frame_coverage():
    cache = FocusLinkerCache.from_nodes([], (3, 20, 32, 32))
    assert cache.manifest()['frame_counts'] == [0, 0, 0]
    assert cache.restore_precise_output(np.empty((0, 4))).shape == (0, 4)


def test_existing_learned_linker_bridge_accepts_focus_cache():
    from types import ModuleType
    from research.lsm_fm_detection.association_bridge import predict_video_with_external_detections

    cache = FocusLinkerCache.from_nodes([node(10, 0, 8), node(20, 2, 12)], (3, 20, 32, 32))
    predictor = ModuleType('fake_official')
    original = lambda *args: None
    predictor._detect_cells_pooled = original

    def predict_video(*args, **kwargs):
        rows = [predictor._detect_cells_pooled(None, t, .9, None) for t in range(3)]
        coords = np.concatenate(rows).astype(float)
        coords[:, 1:] *= cache.downsample
        return coords, []

    predictor.predict_video = predict_video
    points, edges = predict_video_with_external_detections(predictor, None, None, None, None, cache)
    np.testing.assert_array_equal(points, cache.points)
    assert edges == [] and predictor._detect_cells_pooled is original
