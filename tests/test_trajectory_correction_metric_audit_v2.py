from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def contract():
    audit = runpy.run_path(str(ROOT / 'scripts/audit-trajectory-correction-metric-labels-v2.py'))
    helper = runpy.run_path(str(ROOT / 'scripts/score-public-d4-full-movie-v1.py'))
    return audit, helper['load_scorer']()


def test_official_namedtuple_and_summary_contract(contract):
    audit, scorer = contract
    result = scorer.EvaluationResult(10, 2, 3, 1, 0, 0, 20)
    previous = scorer.per_sample_metrics(result, 21., .9)
    base = dict(counts=result._asdict(), summary=scorer.summarise([previous]))
    assert 'edge_tp' not in base['summary']
    audit['assert_baseline_replay'](base, previous, scorer)


def test_count_mismatch_rejected_even_if_summary_unchanged(contract):
    audit, scorer = contract
    result = scorer.EvaluationResult(10, 2, 3, 1, 0, 0, 20)
    previous = scorer.per_sample_metrics(result, 21., .9)
    base = dict(counts=result._asdict(), summary=scorer.summarise([previous]))
    base['counts']['edge_fp'] += 1
    with pytest.raises(AssertionError, match='edge_fp'):
        audit['assert_baseline_replay'](base, previous, scorer)


def test_complete_neutral_sampling_excludes_unknown_and_ambiguous(contract):
    audit, _ = contract
    labels = [dict(label=-1, known_children=1), dict(label=-1, unknown_children=1),
              dict(label=-1, known_children=1, unknown_children=1),
              dict(label=-1, known_children=1, ambiguous_children=1), dict(label=1, known_children=1)]
    parts = [dict(removed=[[1, 2]], added=[]) for _ in labels]
    assert audit['neutral_indices'](labels, parts, complete_only=True) == [0]
    assert audit['neutral_indices'](labels, parts) == [0, 1, 2, 3]
