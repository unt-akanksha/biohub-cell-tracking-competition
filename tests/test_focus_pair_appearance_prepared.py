from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_pair_appearance import projector
from research.focus_pair_appearance_head import blocks, objective
from research.focus_pair_appearance_prepared import Prepared
from research.focus_pair_appearance_resident import Resident
from research.focus_pair_appearance_resident_fit import fit as resident_fit
from research.focus_pair_appearance_head import fit as original_fit

FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_focus_pair_appearance.py')))['fixture']


@pytest.mark.parametrize('arm', ['full', 'lda'])
def test_exact_arrays_and_objective_repeated_reads(arm, tmp_path):
    _, base, app, stats = FIXTURE()
    projection = projector(stats, 'fitting')
    samples = [(base, app)]
    cached = Prepared(samples, projection, arm, tmp_path / arm, 'fitting')
    old = list(blocks(samples, projection, arm))
    for _ in range(2):
        for a, b in zip(old, cached()):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    theta = np.arange(cached.dimension) * .001
    expected = objective(theta, lambda: iter(old), 2.)
    actual = objective(theta, cached, 2.)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    resident = Resident(cached)
    assert resident.mode == 'ram'
    assert not resident.arrays[0].flags.writeable
    replay = objective(theta, resident, 2.)
    assert replay[0] == expected[0]
    np.testing.assert_array_equal(replay[1], expected[1])


def test_role_before_read_and_fold_mismatch(tmp_path):
    with pytest.raises(ValueError, match='Only fitting'):
        Prepared(None, None, None, tmp_path / 'bad', 'diagnostic')
    _, base, app, stats = FIXTURE()
    projection = projector(stats, 'fitting')
    projection['real_pair_counts'][0] += 1
    with pytest.raises(ValueError, match='training pairs'):
        Prepared([(base, app)], projection, 'lda', tmp_path / 'bad', 'fitting')


def test_never_overwrite_cache(tmp_path):
    _, base, app, stats = FIXTURE()
    projection = projector(stats, 'fitting')
    Prepared([(base, app)], projection, 'lda', tmp_path / 'cache', 'fitting')
    with pytest.raises(FileExistsError):
        Prepared([(base, app)], projection, 'lda', tmp_path / 'cache', 'fitting')


@pytest.mark.parametrize('arm', ['full', 'lda'])
def test_resident_optimizer_exact_original_contract(arm, tmp_path):
    _, base, app, stats = FIXTURE()
    projection = projector(stats, 'fitting')
    expected = original_fit([(base, app)], projection, arm, 'fitting')
    actual, execution = resident_fit([(base, app)], projection, arm, tmp_path / arm, 'fitting')
    assert actual == expected
    assert execution['mode'] == 'ram'


def test_resident_fit_role_before_data_access(tmp_path):
    with pytest.raises(ValueError, match='Only fitting'):
        resident_fit(None, None, None, tmp_path / 'bad', 'diagnostic')
