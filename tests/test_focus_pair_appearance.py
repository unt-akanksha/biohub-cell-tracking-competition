import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.optimize import check_grad

from research.focus_candidate_ranker import pack, candidate_arrays, validate
from research.focus_balanced_candidate_ranker import objective as old_objective
from research.focus_pair_appearance import descriptors, empty_stats, accumulate, merge_stats, projector, transform
from research.focus_pair_appearance_head import blocks, objective, fit, metrics

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = runpy.run_path(str(ROOT / 'tests/test_focus_candidate_ranker.py'))


def fixture():
    packet = FIXTURE['sample']()
    base = pack(packet, FIXTURE['PARAMETERS'], 'fitting')
    appearance = descriptors(packet, np.flatnonzero(packet['labels'] >= 0))
    stats = empty_stats()
    accumulate(stats, appearance, base, 'fitting')
    return packet, base, appearance, stats


def test_complete_descriptor_and_cosine_identity():
    packet, base, x, _ = fixture()
    assert x.shape == (12, 64) and x.dtype == np.float32
    np.testing.assert_allclose(x[:, 32:].sum(axis=1), base['features'][:, 7], atol=1e-6)
    assert np.all(x[base['null_rows']] == 0)
    assert descriptors(packet).shape == (16, 64)  # Label-free inference includes unknown targets.
    packet.pop('labels')
    assert descriptors(packet).shape == (16, 64)


def test_class_counts_exclude_nulls_and_unknowns():
    _, base, x, stats = fixture()
    np.testing.assert_array_equal(stats['count'], [7, 2])
    real = np.ones(len(x), bool)
    real[base['null_rows']] = False
    np.testing.assert_allclose(stats['sums'].sum(axis=0), x[real].astype(float).sum(axis=0))
    np.testing.assert_allclose(stats['seconds'].sum(axis=0), x[real].astype(float).T @ x[real].astype(float), atol=1e-12)


def test_streamed_moments_and_train_only_merge():
    _, _, _, stats = fixture()
    combined = merge_stats([stats, stats], 'fitting')
    for key in stats:
        np.testing.assert_allclose(combined[key], 2 * stats[key])
    before = projector(stats, 'fitting')
    held_out = empty_stats()
    held_out['sums'][:] = 1000  # Not supplied to the training-only merge.
    assert projector(merge_stats([stats], 'fitting'), 'fitting') == before


def test_projection_roundtrip_zero_null_and_unit_variance():
    _, base, x, stats = fixture()
    model = projector(stats, 'fitting')
    restored = json.loads(json.dumps(model))
    full = transform(x, base['null_rows'], model, 'full')
    lda = transform(x, base['null_rows'], model, 'lda')
    assert full.shape == (12, 64) and lda.shape == (12, 1)
    assert np.all(full[base['null_rows']] == 0) and np.all(lda[base['null_rows']] == 0)
    np.testing.assert_array_equal(lda, transform(x, base['null_rows'], restored, 'lda'))
    keep = np.ones(len(x), bool)
    keep[base['null_rows']] = False
    assert lda[keep].mean() == pytest.approx(0., abs=1e-10)
    assert lda[keep].var() == pytest.approx(1., abs=1e-8)
    assert 0 < model['retained_rank'] < 64  # Handles singular within-class covariance.


def test_roles_rejected_before_access():
    with pytest.raises(ValueError, match='Only fitting'):
        accumulate({}, None, {}, 'diagnostic')
    with pytest.raises(ValueError, match='Only fitting'):
        projector({}, 'diagnostic')
    with pytest.raises(ValueError, match='Only fitting'):
        merge_stats([], 'diagnostic')
    with pytest.raises(ValueError, match='Only fitting'):
        fit([], {}, 'full', 'diagnostic')


def test_no_fallback_for_missing_class_or_zero_signal():
    with pytest.raises(ValueError, match='At least two'):
        projector(empty_stats(), 'fitting')
    stats = empty_stats()
    stats['count'][:] = 3
    with pytest.raises(ValueError, match='covariance'):
        projector(stats, 'fitting')


@pytest.mark.parametrize('arm', ['full', 'lda'])
def test_generalized_softmax_gradient_and_block_invariance(arm):
    _, base, x, stats = fixture()
    projection = projector(stats, 'fitting')
    samples = [(base, x)]
    dim = 72 if arm == 'full' else 9
    provider = lambda: blocks(samples, projection, arm)
    theta = np.zeros(dim)
    weight = np.sqrt(2)
    assert check_grad(lambda p: objective(p, provider, weight)[0],
                      lambda p: objective(p, provider, weight)[1], theta) < 5e-4
    normal = objective(theta, provider, weight)
    single = objective(theta, lambda: blocks(samples, projection, arm, group_block=1), weight)
    assert normal[0] == pytest.approx(single[0], abs=1e-12)
    np.testing.assert_allclose(normal[1], single[1], atol=1e-10)


def test_original_eight_feature_loss_exact_reduction():
    _, base, x, _ = fixture()
    theta = np.array([.2, .01, -.02, .03, -.01, -.02, -.03, .4])
    weights = np.where(base['present'], 1., np.sqrt(2))
    expected = old_objective(theta, base, validate(base), weights)
    actual = objective(theta, lambda: blocks([(base, x)], None, 'base'), np.sqrt(2))
    assert actual[0] == expected[0]
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)


@pytest.mark.parametrize('arm', ['full', 'lda'])
def test_smoke_head_roundtrip_and_projection_count_guard(arm):
    _, base, x, stats = fixture()
    projection = projector(stats, 'fitting')
    samples = [(base, x)]
    model = fit(samples, projection, arm, 'fitting')
    assert metrics(samples, model) == metrics(samples, json.loads(json.dumps(model)))
    assert model['objective'] <= model['initial_objective']
    wrong = dict(projection, real_pair_counts=[700, 200])
    with pytest.raises(ValueError, match='exact same fitting'):
        fit(samples, wrong, arm, 'fitting')


def test_empty_source_has_no_real_pair_descriptors():
    packet = FIXTURE['sample']()
    packet['source_coords'] = np.empty((0, 3))
    packet['source_features'] = np.empty((0, 32))
    packet['labels'] = np.array([0, 0, 0, -1], np.int64)
    base = pack(packet, FIXTURE['PARAMETERS'], 'fitting')
    appearance = descriptors(packet, np.flatnonzero(packet['labels'] >= 0))
    stats = empty_stats()
    accumulate(stats, appearance, base, 'fitting')
    assert appearance.shape == (3, 64) and not appearance.any()
    assert not stats['count'].any()
