from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_candidate_ranker import candidate_arrays
from research.focus_pair_appearance import projector, descriptors, transform
from research.focus_pair_appearance_inference import predict, score_blocks, validate_model

FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_focus_pair_appearance.py')))['fixture']


def fixture(arm):
    packet, _, _, stats = FIXTURE()
    packet['source_indices'] = np.arange(3, dtype=np.int64)
    model = dict(arm=arm, role='fitting', ridge=1., null_logit=-4.5,
                 squared_correction_upper_bound=.5, projection=projector(stats, 'fitting'),
                 theta=(np.arange(72 if arm == 'full' else 9) * .002).tolist())
    parameters = dict(mean_um=[0., 0., 0.], variance_um2=[1., 1., 1.])
    return packet, model, parameters


@pytest.mark.parametrize('arm', ['full', 'lda'])
@pytest.mark.parametrize('size', [1, 2, 32, 64])
def test_complete_dense_equivalence_without_labels(arm, size):
    packet, model, parameters = fixture(arm)
    packet.pop('labels')
    original = deepcopy(packet)
    base = candidate_arrays(packet, parameters)
    extra = transform(descriptors(packet), base['null_rows'], model['projection'], arm)
    scores = (base['offset'] + np.column_stack([base['features'], extra]) @ np.asarray(model['theta'])).reshape(4, 4)
    blocks = list(score_blocks(packet, parameters, model, size))
    np.testing.assert_allclose(np.concatenate([b['scores'] for b in blocks]), scores, atol=1e-10, rtol=1e-10)
    output = predict(packet, parameters, model, size)
    expected = scores.argmax(axis=1)
    np.testing.assert_array_equal(output['source_local_indices'], np.where(expected == 3, -1, expected))
    np.testing.assert_array_equal(output['target_indices'], packet['target_indices'])
    assert len(output['target_indices']) == 4  # Includes the originally unknown target.
    for key in packet:
        np.testing.assert_array_equal(packet[key], original[key])


def test_null_only_and_empty_target_frames():
    packet, model, parameters = fixture('full')
    packet.pop('labels')
    packet.update(source_indices=np.empty(0, np.int64), source_coords=np.empty((0, 3)),
                  source_features=np.empty((0, 32)))
    output = predict(packet, parameters, model)
    np.testing.assert_array_equal(output['source_indices'], np.full(4, -1))
    np.testing.assert_array_equal(output['null_probability'], np.ones(4))
    packet.update(target_indices=np.empty(0, np.int64), target_coords=np.empty((0, 3)),
                  target_features=np.empty((0, 32)), backward_um=np.empty((0, 3)))
    assert len(predict(packet, parameters, model)['target_indices']) == 0
    model['theta'][4] = .51
    with pytest.raises(ValueError):
        predict(packet, parameters, model)  # Empty frames do not bypass model validation.


def test_exact_tie_prefers_first_source_even_against_null():
    packet, model, parameters = fixture('full')
    packet['source_coords'][:] = 0
    packet['target_coords'][:] = 0
    model['theta'] = [0.] * 72
    model['theta'][0] = -4.5
    out = predict(packet, parameters, model, 1)
    np.testing.assert_array_equal(out['source_indices'], np.zeros(4, np.int64))


@pytest.mark.parametrize('size', [0, 65, True, 1.5])
def test_bad_block_sizes_rejected(size):
    packet, model, parameters = fixture('lda')
    with pytest.raises(ValueError):
        predict(packet, parameters, model, size)


def test_mismatched_or_overlapping_identity_rejected():
    packet, model, parameters = fixture('lda')
    packet['target_indices'][0] = 0
    with pytest.raises(ValueError, match='identities'):
        predict(packet, parameters, model)


def test_nonfinite_coefficients_rejected():
    _, model, _ = fixture('full')
    model['theta'][0] = np.inf
    with pytest.raises(ValueError):
        validate_model(model)
