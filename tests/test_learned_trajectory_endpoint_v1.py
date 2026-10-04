import runpy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'research/learned_trajectory_endpoint_v1.py'))
FIX = runpy.run_path(str(ROOT / 'tests/test_public_reciprocal_endpoint_v1.py'))['fixture']


def training():
    rng = np.random.default_rng(1827)
    velocity = rng.normal(size=(300, 1, 3))
    positions = velocity * np.arange(6)[None, :, None] + rng.normal(scale=.1, size=(300, 6, 3))
    return positions, np.arange(300) // 10


def test_small_fit_finite_replay_and_translation_invariance():
    windows, movies = training(); model = M['fit'](windows, movies)
    score = M['scores'](model, windows)
    assert np.isfinite(score).all() and (score >= 0).all()
    np.testing.assert_allclose(score, M['scores'](model, windows + 100), atol=1e-10)
    threshold = M['calibrate'](model, windows)
    assert (score <= threshold).mean() >= .99
    np.testing.assert_array_equal(model['coef'], M['fit'](windows, movies)['coef'])


def test_reconnect_retains_nodes_and_respects_synthetic_provenance():
    windows, movies = training(); model = M['fit'](windows, movies)
    threshold = M['calibrate'](model, windows)
    graph, coords, raw = FIX()
    out, details = M['reconnect'](graph, coords, raw, range(6), model, threshold)
    assert details['added_edges'] == 1 and out['nodes'] == graph['nodes']
    assert M['reconnect'](graph, coords, raw, [0, 1, 2, 4, 5], model, threshold)[1]['added_edges'] == 0


def test_reconnection_inference_has_no_annotation_argument():
    import inspect
    assert list(inspect.signature(M['reconnect']).parameters) == [
        'final', 'coords', 'raw_edges', 'detector_ids', 'model', 'threshold']
