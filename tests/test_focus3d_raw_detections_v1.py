import json
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / 'scripts/build-focus3d-raw-detections-v1.py'))


def test_notebook_omits_labels_and_physical_postprocessing():
    notebook = MODULE['build']()
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    assert 'require_tracks=True' not in code
    assert 'postprocess_tracks(' not in code
    assert "raise RuntimeError(f'{n_failed} FOCUS frames failed')" in code
    assert "meta['movie_shape'] = list(volume.shape)" in code
    assert 'raw_detections=records' in code
    assert 'f"{stem}.geff"' not in code
    assert 'f"{stem}.npz"' in code


def test_saver_preserves_raw_coordinates_and_empty_frames(tmp_path):
    namespace = {'np': np, 'Path': Path, 'json': json, 'predict_dir': str(tmp_path)}
    exec(MODULE['SAVE_RAW'], namespace)
    coords = [[0, 1.25, 4.5, 6.75], [2, 2.25, 6.5, 7.75]]
    meta = {'movie_shape': [4, 16, 32, 32], 'scale_um': [1.625, .40625, .40625]}
    save = namespace['_physical_pp_and_save']
    save(coords, [], meta, 'sample', 0, 0)
    with np.load(tmp_path / 'sample.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['coords'], coords)
        from research.focus_linker_cache import FocusLinkerCache
        cache = FocusLinkerCache.from_nodes(
            [dict(node_id=i, **dict(zip(('t', 'z', 'y', 'x'), row)))
             for i, row in enumerate(data['coords'])], data['movie_shape'],
            coordinate_source='raw instance centroids')
        np.testing.assert_array_equal(cache.precise_output_coords(), coords)
        assert cache.manifest()['coordinate_source'] == 'raw instance centroids'
    receipt = json.loads((tmp_path / 'sample.json').read_text())
    assert receipt['frame_counts'] == [1, 0, 1, 0]
    assert not receipt['postprocessing_applied']
    with pytest.raises(RuntimeError, match='bounds'):
        save([[0, -1, 2, 3]], [], meta, 'invalid', 0, 0)
    assert not (tmp_path / 'invalid.npz').exists()
