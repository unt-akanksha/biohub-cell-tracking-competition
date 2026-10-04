import hashlib

import numpy as np
import pytest

from research.focus_linker_runtime import install, load_raw_cache


def fixture(tmp_path):
    coords = np.asarray([[0, 1.25, 8.5, 8.5], [1, 2.25, 12.5, 12.5]])
    path = tmp_path / 'movie.npz'
    np.savez_compressed(path, coords=coords, movie_shape=[2, 16, 32, 32], scale_um=[1, 1, 1])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(status='completed', ground_truth_opened=False, raw_detections=[
        dict(stem='movie', sha256=digest, failed_frames=0, postprocessing_applied=False)])
    namespace = {'_detect_cells_pooled': lambda *args: None}
    def predict_video(model, ds_path, device, cfg, max_frames=None, downsample=(1, 4, 4)):
        coords = np.concatenate([namespace['_detect_cells_pooled'](None, t, .99, None) for t in range(2)]).astype(float)
        coords[:, 1:] *= downsample
        return coords, [(0, 1, .9, 1.)]
    namespace['predict_video'] = predict_video
    return coords, digest, manifest, namespace


def test_real_wrapper_restores_nodes_and_preserves_edges(tmp_path):
    coords, _, manifest, namespace = fixture(tmp_path)
    detector = namespace['_detect_cells_pooled']
    predict = install(namespace, tmp_path, manifest, lambda path: (2, 16, 32, 32))
    actual, edges = predict(None, tmp_path / 'movie.zarr', None, None)
    np.testing.assert_array_equal(actual, coords)
    assert edges == [(0, 1, .9, 1.)]
    assert namespace['_detect_cells_pooled'] is detector
    with pytest.raises(ValueError, match='complete movies'):
        predict(None, tmp_path / 'movie.zarr', None, None, max_frames=1)
    with pytest.raises(ValueError, match='outside frozen'):
        predict(None, tmp_path / 'other.zarr', None, None)


def test_checkpoint_hash_and_shape_are_checked_before_use(tmp_path):
    _, digest, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='hash'):
        load_raw_cache(tmp_path, 'movie', '0' * 64, (2, 16, 32, 32))
    with pytest.raises(ValueError, match='shape'):
        load_raw_cache(tmp_path, 'movie', digest, (3, 16, 32, 32))


def test_official_cli_bare_movie_stem_uses_zarr_shape_path(tmp_path):
    coords, _, manifest, namespace = fixture(tmp_path)
    observed = []
    def shape_reader(path):
        observed.append(path)
        assert path == tmp_path / 'movie.zarr'
        return (2, 16, 32, 32)
    predict = install(namespace, tmp_path, manifest, shape_reader)
    actual, _ = predict(None, tmp_path / 'movie', None, None)
    np.testing.assert_array_equal(actual, coords)
    assert observed == [tmp_path / 'movie.zarr']


def test_exception_restores_original_detector(tmp_path):
    _, _, manifest, namespace = fixture(tmp_path)
    detector = namespace['_detect_cells_pooled']
    def failure(model, ds_path, device, cfg):
        raise RuntimeError('inference failure')
    namespace['predict_video'] = failure
    predict = install(namespace, tmp_path, manifest, lambda path: (2, 16, 32, 32))
    with pytest.raises(RuntimeError, match='inference failure'):
        predict(None, tmp_path / 'movie.zarr', None, None)
    assert namespace['_detect_cells_pooled'] is detector
