import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fork16_evaluation',ROOT/'scripts/evaluate-trajectory-event-fork16-held-v1.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixtures():
    def row(score,n):
        return dict(n=n,n_adj=n,score=np.float64(score),edge_jaccard=score-.01,
                    adj_edge_jaccard=score-.01,division_tp=1,division_fp=1,division_fn=1,
                    division_jaccard=1/3)
    summaries={a:row(s,2) for a,s in [('submitted8',.90),('untrained16',.91),('learned16',.92)]}
    movies={a:{m:row(s['score'],1) for m in ('a','b')} for a,s in summaries.items()}
    return summaries,movies


def test_quality_pass_is_json_serializable():
    summaries,movies=fixtures()
    checks,regressions=module.quality_checks(summaries,movies,['a','b'])
    assert all(checks.values()) and not regressions
    assert all(type(v) is bool for v in checks.values())
    json.dumps(checks,allow_nan=False)


def test_one_regression_not_hidden_by_pooled_gain():
    summaries,movies=fixtures();movies['learned16']['b']['score']=.89
    checks,regressions=module.quality_checks(summaries,movies,['a','b'])
    assert checks['pooled_score_improves'] and not checks['every_movie_nonregressing']
    assert regressions['b']==pytest.approx(-.01)


def test_vocabulary_gain_alone_does_not_qualify_learning():
    summaries,movies=fixtures();summaries['untrained16']['score']=.93
    checks,_=module.quality_checks(summaries,movies,['a','b'])
    assert checks['pooled_score_improves'] and not checks['learning_beats_untrained_control']


@pytest.mark.parametrize('location',['aggregate','movie'])
def test_undefined_or_skipped_scores_fail(location):
    summaries,movies=fixtures()
    row=summaries['learned16'] if location=='aggregate' else movies['learned16']['a']
    row['score']=float('nan');row['n_adj']=0
    checks,_=module.quality_checks(summaries,movies,['a','b'])
    assert not checks['finite_complete_scoring']


def test_missing_movie_rejected():
    summaries,movies=fixtures();del movies['learned16']['b']
    with pytest.raises(AssertionError):module.quality_checks(summaries,movies,['a','b'])


def test_empty_division_term_not_a_false_failure_but_new_false_division_fails():
    summaries,movies=fixtures()
    for s in summaries.values():
        s.update(division_tp=0,division_fp=0,division_fn=0,division_jaccard=float('nan'))
    checks,_=module.quality_checks(summaries,movies,['a','b'])
    assert checks['division_jaccard_nonregressing']
    summaries['learned16'].update(division_fp=1,division_jaccard=0.)
    checks,_=module.quality_checks(summaries,movies,['a','b'])
    assert not checks['division_jaccard_nonregressing']


def test_real_frozen_fit_contracts_without_opening_truth():
    assert set(module.fixed_fit_contracts(ROOT))=={'44b6','6bba'}
