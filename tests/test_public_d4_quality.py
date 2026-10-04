import copy
import pytest
from research.public_d4_full_movie import STEMS
from research.public_d4_quality import compare


def inputs():
    rows={a:[{'stem':s} for s in STEMS] for a in ('original','corrected')}
    summaries={a:{k:.8+i*.01 for k in ('score','edge_jaccard','adj_edge_jaccard')}
               for i,a in enumerate(('original','corrected'))}
    embryos={a:{e:{'score':.8+i*.01} for e in ('44b6','6bba')}
             for i,a in enumerate(('original','corrected'))}
    movies={a:{s:{'score':.8+i*.01} for s in STEMS}
            for i,a in enumerate(('original','corrected'))}
    return rows,summaries,embryos,movies


def test_diagnostic_pass_never_authorizes_submission():
    result=compare(*inputs())
    assert result['diagnostic_gate_passed']
    assert result['authorized_for_submission'] is False
    assert result['independently_held_out'] is False


@pytest.mark.parametrize('failure',['pooled','raw_edge','embryo','movie'])
def test_no_pooled_gain_can_hide_regressions(failure):
    rows,summaries,embryos,movies=inputs()
    if failure=='pooled': summaries['corrected']['score']=.8
    if failure=='raw_edge': summaries['corrected']['edge_jaccard']=.799
    if failure=='embryo': embryos['corrected']['6bba']['score']=.799
    if failure=='movie': movies['corrected'][STEMS[-1]]['score']=.799
    assert not compare(rows,summaries,embryos,movies)['diagnostic_gate_passed']


def test_missing_movie_and_nonfinite_rejected():
    args=inputs();args[0]['corrected'].pop()
    with pytest.raises(ValueError): compare(*args)
    args=inputs();args[1]['corrected']['score']=float('nan')
    with pytest.raises(ValueError): compare(*args)
