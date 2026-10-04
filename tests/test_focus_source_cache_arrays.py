from pathlib import Path
import runpy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/verify-focus-source-cache.py'))


def test_raw_centroid_coverage_preserves_empty_frames():
    coords = np.asarray([[0, 1.2, 3.4, 5.6], [2, 63.9, 255.9, 255.9]], dtype=np.float32)
    assert M['validate_arrays'](coords, [3, 64, 256, 256], [1.625, .40625, .40625], 3) == [1, 0, 1]


@pytest.mark.parametrize('coords', [
    [[0, float('nan'), 2, 3]], [[0, -1, 2, 3]], [[0, 64, 2, 3]],
    [[3, 1, 2, 3]], [[.5, 1, 2, 3]], [[0, 1, 256, 3]], [[0, 1, 2, 256]]])
def test_invalid_centroids_fail_without_clipping(coords):
    with pytest.raises(ValueError, match='centroids'):
        M['validate_arrays'](np.asarray(coords, dtype=np.float32), [3, 64, 256, 256], [1.625, .40625, .40625], 3)


def test_incorrect_geometry_or_precision_rejected():
    coords = np.asarray([[0, 1, 2, 3]], dtype=np.float32)
    for array, shape, scale in [(coords.astype(np.float64), [3,64,256,256], [1.625,.40625,.40625]),
                                (coords, [100,64,256,256], [1.625,.40625,.40625]),
                                (coords, [3,64,256,256], [1,1,1])]:
        with pytest.raises(ValueError):
            M['validate_arrays'](array, shape, scale, 3)
