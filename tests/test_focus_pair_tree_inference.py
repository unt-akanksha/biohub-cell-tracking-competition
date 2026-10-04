from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_candidate_ranker import candidate_arrays
from research.focus_pair_appearance import descriptors, transform
from research.focus_pair_tree import portable_predict, VERSION, SETTINGS, ROUNDS
from research.focus_pair_tree_inference import predict, score_blocks

FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_focus_pair_appearance_inference.py')))['fixture']


class Residual:
    def __init__(self):
        tree = dict(nodeid=0, split='f7', split_condition=.25, yes=1, no=2, missing=1,
                    children=[dict(nodeid=1, leaf=.025), dict(nodeid=2, leaf=-.025)])
        self.model = dict(version=VERSION, settings=SETTINGS, trees=[deepcopy(tree) for _ in range(ROUNDS)])

    def predict(self, features):
        return portable_predict(self.model, features)


@pytest.mark.parametrize('size', [1, 2, 32, 64])
def test_complete_label_free_equivalence_and_identity(size):
    packet, baseline, parameters = FIXTURE('full')
    packet.pop('labels')
    original, residual = deepcopy(packet), Residual()
    base = candidate_arrays(packet, parameters)
    app = transform(descriptors(packet), base['null_rows'], baseline['projection'], 'full')
    features = np.column_stack([base['features'], app])
    score = (base['offset']+features @ np.asarray(baseline['theta'])+residual.predict(features)).reshape(4, 4)
    score -= (score[:, -1]+4.5)[:, None]
    actual = np.concatenate([b['scores'] for b in score_blocks(packet, parameters, baseline, residual, size)])
    np.testing.assert_allclose(actual, score, rtol=0, atol=1e-10)
    result = predict(packet, parameters, baseline, residual, size)
    expected = score.argmax(axis=1)
    np.testing.assert_array_equal(result['source_local_indices'], np.where(expected == 3, -1, expected))
    np.testing.assert_array_equal(result['target_indices'], original['target_indices'])
    assert len(result['target_indices']) == 4  # Includes originally unknown target.
    for key in original:
        np.testing.assert_array_equal(packet[key], original[key])


def test_empty_sources_targets_and_duplicate_ids():
    packet, baseline, parameters = FIXTURE('full')
    residual = Residual()
    bad = deepcopy(packet)
    bad['target_indices'][0] = 0
    with pytest.raises(ValueError, match='identities'):
        predict(bad, parameters, baseline, residual)
    packet.update(source_indices=np.empty(0, np.int64), source_coords=np.empty((0, 3)), source_features=np.empty((0, 32)))
    result = predict(packet, parameters, baseline, residual)
    np.testing.assert_array_equal(result['source_indices'], np.full(4, -1))
    np.testing.assert_array_equal(result['null_probability'], np.ones(4))
    packet.update(target_indices=np.empty(0, np.int64), target_coords=np.empty((0, 3)),
                  target_features=np.empty((0, 32)), backward_um=np.empty((0, 3)))
    assert len(predict(packet, parameters, baseline, residual)['target_indices']) == 0


@pytest.mark.parametrize('size', [0, 65, True, 1.5])
def test_bad_block_sizes_rejected(size):
    packet, baseline, parameters = FIXTURE('full')
    with pytest.raises(ValueError):
        predict(packet, parameters, baseline, Residual(), size)


def test_wrong_baseline_dimension_rejected():
    packet, baseline, parameters = FIXTURE('lda')
    with pytest.raises(ValueError, match='full72D'):
        predict(packet, parameters, baseline, Residual())
