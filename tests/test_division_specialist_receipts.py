import copy
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN = runpy.run_path(str(ROOT/'scripts/summarize-division-specialist-training.py'))
SCORE = runpy.run_path(str(ROOT/'scripts/summarize-division-specialist-selection.py'))


def test_real_division_exposure_and_exact_sampler_are_required():
    result,terminal,split = runpy.run_path(str(ROOT/'tests/test_known_null_training_receipts.py'))['receipt']()
    result['identity'].update(initialization_sha256=TRAIN['INITIAL'],
        loss='division_balanced_with_annotated_missing_parent_null_v1',
        division_specialist=dict(version=1,division_window_mass=.5),
        division_sampling=dict(division_windows=114,ordinary_windows=11509,division_mass=.5,replacement=True,
            weights_sha256='05390c6f1050f5d6e4a269f9e975ebaf104806d49f68a8d8afc854535b85c106'))
    for row in result['history']:
        row.update(division_columns=2,correct_division_columns=1)
    result['division_supervised_total'] = 20
    assert TRAIN['verify'](result,terminal,split,100)['division_columns'] == 20
    for change in ('sampler','exposure'):
        bad = copy.deepcopy(result)
        if change == 'sampler':
            bad['identity']['division_sampling']['division_mass'] = .8
        else:
            bad['division_supervised_total'] = 0
        with pytest.raises(ValueError):
            TRAIN['verify'](bad,terminal,split,100)


def test_candidate_comparison_uses_improved_parent_not_only_old_baseline():
    candidate,manifest,training,native,causal = runpy.run_path(str(ROOT/'tests/test_known_null_selection_comparison.py'))['inputs']()
    sampling = dict(division_mass=.5)
    manifest.update(division_specialist_training=dict(version=1,division_window_mass=.5),division_sampling=sampling)
    training.update(division_columns=10,division_sampling=sampling)
    parent = copy.deepcopy(native)
    parent.update(checkpoint_sha256=SCORE['INITIAL'],summary=dict(score=.85))
    report = SCORE['compare'](candidate,manifest,training,native,causal,parent)
    assert report['delta_vs_native'] > 0 and report['delta_vs_parent'] < 0
    assert report['authorized_for_submission'] is False
    parent['per_movie'][0]['node_recall'] = .1
    with pytest.raises(ValueError):
        SCORE['compare'](candidate,manifest,training,native,causal,parent)
