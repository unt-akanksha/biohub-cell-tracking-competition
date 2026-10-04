from copy import deepcopy
from pathlib import Path
import json
import runpy
import warnings
import pytest
from research.dense_warp_movie_quality_v1 import compare

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    base=dict(score=.90,edge_jaccard=.91,adj_edge_jaccard=.89,n=4,n_adj=4)
    summaries={a:deepcopy(base) for a in ('original','warp44','warp6')}
    embryos={a:{e:deepcopy(base) for e in ('44b6','6bba')} for a in summaries}
    movies={a:{s:deepcopy(base) for s in ('44b6_a','44b6_b','6bba_a','6bba_b')} for a in summaries}
    return summaries,embryos,movies


def test_neutral_is_not_improvement():
    result=compare(*fixture())
    assert not any(r['diagnostic_pass'] for r in result.values())


def test_pooled_gain_cannot_hide_movie_regression():
    summaries,embryos,movies=fixture()
    summaries['warp44'].update(score=.92,edge_jaccard=.93)
    movies['warp44']['44b6_a']['score']=.89
    assert not compare(summaries,embryos,movies)['warp44']['diagnostic_pass']


def test_invalid_or_skipped_metric_rows_rejected():
    summaries,embryos,movies=fixture();summaries['warp6']['n_adj']=3
    with pytest.raises(ValueError):compare(summaries,embryos,movies)
    summaries['warp6']['n_adj']=4;embryos['warp6']['6bba']['score']=float('nan')
    with pytest.raises(ValueError):compare(summaries,embryos,movies)


def test_scorer_drops_no_division_term_and_micro_averages_edges():
    helper=runpy.run_path(str(ROOT/'scripts/score-public-d4-full-movie-v1.py'))
    scorer=helper['load_scorer']()
    large=scorer.EvaluationResult(90,0,10,0,0,0,100)
    small=scorer.EvaluationResult(1,0,9,0,0,0,10)
    rows=[scorer.per_sample_metrics(large,100,1),scorer.per_sample_metrics(small,10,1)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');result=scorer.summarise(rows)
    assert result['edge_jaccard']==pytest.approx(91/110)
    assert result['score']==pytest.approx(91/110)
    assert result['score']!=pytest.approx(.5)


def test_new_score_entry_rejects_smoke_before_truth_access(tmp_path):
    namespace=runpy.run_path(str(ROOT/'scripts/score-dense-warp-movies-v1.py'))
    main=namespace['main'];main.__globals__['ROOT']=tmp_path
    bundle=tmp_path/'.biohub/cache/dense-warp-movie-v1-bundle';bundle.mkdir(parents=True)
    (bundle/'CONTRACT.json').write_text('{}')
    predictions=tmp_path/'.biohub/cache/dense-warp-movie-v1-full-output';predictions.mkdir()
    (predictions/'result.json').write_text(json.dumps(dict(status='functionality_passed',mode='smoke')))
    # There is deliberately no truth folder in this fixture.
    with pytest.raises(AssertionError):main()
