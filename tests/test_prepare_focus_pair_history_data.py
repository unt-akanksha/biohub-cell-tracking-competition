from pathlib import Path
import runpy

import numpy as np

from research.focus_pair_history_head import pack

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/prepare-focus-pair-history-data.py'))


def test_independent_full_known_features_and_unknown_exclusion():
    packet = dict(source_frame=np.array(1, np.int64), source_coords=np.array([[2., 3., 4.], [5., 6., 7.]]),
        target_coords=np.array([[3., 4., 5.], [5., 5., 5.], [9., 10., 11.]]),
        source_indices=np.array([10, 11], np.int64), target_indices=np.array([20, 21, 22], np.int64),
        source_features=np.ones((2, 32)), target_features=np.ones((3, 32)),
        backward_um=np.zeros((3, 3)), labels=np.array([0, -1, 2], np.int64))
    previous = dict(source_frame=np.array(0, np.int64), target_coords=packet['source_coords'][::-1].copy(),
        target_indices=np.array([11, 10], np.int64), backward_um=np.array([[1., 2., 3.], [2., 3., 4.]]))
    parameters = dict(mean_um=[.1, .2, .3], variance_um2=[1., 2., 3.])
    actual = pack(packet, previous, parameters, 'fitting')
    reference = SCRIPT['independent_history'](packet, previous, parameters)
    np.testing.assert_allclose(actual['features'][:,8:], reference, atol=1e-12)
    np.testing.assert_array_equal(actual['target_indices'], [20, 22])
    assert len(actual['offset']) == 6
    packet['source_frame'] = np.array(0, np.int64)
    assert not SCRIPT['independent_history'](packet, None, parameters).any()
