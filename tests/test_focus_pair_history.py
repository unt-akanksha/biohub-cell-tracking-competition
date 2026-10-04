import copy

import numpy as np
import pytest

from research.focus_pair_history import candidate_history, previous_source_motion
from research.focus_candidate_ranker import SCALE


PARAMS = dict(mean_um=[0., 0., 0.], variance_um2=[1., 1., 1.])


def example():
    source = np.array([[2., 5., 8.], [5., 8., 11.]])
    velocity = np.array([1., 2., 3.])
    packet = dict(source_frame=np.array(1, np.int64), source_coords=source,
        target_coords=source+velocity, source_indices=np.array([10, 11], np.int64),
        target_indices=np.array([20, 21], np.int64), source_features=np.ones((2, 32)),
        target_features=np.ones((2, 32)), backward_um=np.tile(-velocity*SCALE, (2, 1)))
    previous = dict(source_frame=np.array(0, np.int64), target_coords=source.copy(),
        target_indices=packet['source_indices'].copy(), backward_um=packet['backward_um'].copy())
    return packet, previous


def test_backward_sign_and_anisotropic_scaling():
    packet, previous = example()
    features = candidate_history(packet, previous, PARAMS)['features'].reshape(2, 3, 7)
    np.testing.assert_array_equal(features[[0, 1], [0, 1], 1:], 0)
    np.testing.assert_array_equal(features[:, 2], 0)
    np.testing.assert_array_equal(features[:, :2, 0], 1)
    np.testing.assert_allclose(features[0, 1, 1:4], np.array([3., 3., 3.])*SCALE)


def test_previous_rows_join_by_original_id():
    packet, previous = example()
    previous['backward_um'][1] *= 2
    expected = previous_source_motion(packet, previous)[0]
    for key in ('target_coords', 'target_indices', 'backward_um'):
        previous[key] = previous[key][::-1].copy()
    np.testing.assert_array_equal(previous_source_motion(packet, previous)[0], expected)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'geometry', 'frame', 'nonfinite'])
def test_invalid_history_rejected(mutation):
    packet, previous = example()
    if mutation == 'duplicate': previous['target_indices'][1] = 10
    if mutation == 'missing': previous['target_indices'][1] = 99
    if mutation == 'geometry': previous['target_coords'][1, 0] += .01
    if mutation == 'frame': previous['source_frame'] = np.array(1, np.int64)
    if mutation == 'nonfinite': previous['backward_um'][0, 0] = np.nan
    with pytest.raises(ValueError):
        candidate_history(packet, previous, PARAMS)


def test_only_first_transition_may_lack_history():
    packet, previous = example()
    with pytest.raises(ValueError): candidate_history(packet, None, PARAMS)
    packet['source_frame'] = np.array(0, np.int64)
    output = candidate_history(packet, None, PARAMS)
    assert not output['source_history_available'].any()
    assert not output['features'].any()
    with pytest.raises(ValueError): candidate_history(packet, previous, PARAMS)


def test_labels_not_read_inputs_unchanged_and_blocks_exact():
    class NoLabels(dict):
        def __getitem__(self, key):
            if key == 'labels': raise AssertionError('Labels must never be accessed')
            return super().__getitem__(key)
    packet, previous = example()
    saved = copy.deepcopy((packet, previous))
    full = candidate_history(NoLabels(packet), NoLabels(previous), PARAMS)['features']
    parts = [candidate_history(NoLabels(packet), NoLabels(previous), PARAMS,
              np.array([i], np.int64))['features'] for i in range(2)]
    np.testing.assert_array_equal(full, np.concatenate(parts))
    for before, after in zip(saved, (packet, previous)):
        for key in before: np.testing.assert_array_equal(before[key], after[key])


def test_bad_columns_and_cross_frame_identity_rejected():
    packet, previous = example()
    for columns in (np.array([0, 0], np.int64), np.array([2], np.int64), np.array([0.])):
        with pytest.raises(ValueError): candidate_history(packet, previous, PARAMS, columns)
    packet['target_indices'][0] = 10
    with pytest.raises(ValueError): candidate_history(packet, previous, PARAMS)


def test_empty_source_and_target_are_supported():
    packet, previous = example()
    for side in ('source', 'target'):
        for suffix in ('coords', 'indices', 'features'):
            packet[side+'_'+suffix] = packet[side+'_'+suffix][:0]
    packet['backward_um'] = packet['backward_um'][:0]
    for key in ('target_coords', 'target_indices', 'backward_um'):
        previous[key] = previous[key][:0]
    result = candidate_history(packet, previous, PARAMS)
    assert result['features'].shape == (0, 7)
