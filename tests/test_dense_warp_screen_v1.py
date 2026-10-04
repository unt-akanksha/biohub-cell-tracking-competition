from copy import deepcopy
import pytest
from research.dense_warp_screen_v1 import checked_gate


def metrics(correct=3, ce=1.):
    return dict(correct=correct,total=5,ce=ce,per_movie={'44b6_a':dict(correct=correct,total=5,annotated=7)})


def test_requires_real_ranking_and_loss_gain():
    assert checked_gate(metrics(),metrics(4,.8))
    assert not checked_gate(metrics(),metrics(3,.8))
    assert not checked_gate(metrics(),metrics(4,1.1))


def test_rejects_silent_denominator_change():
    after=metrics(4,.8);after['per_movie']['44b6_a']['annotated']=8
    with pytest.raises(ValueError):checked_gate(metrics(),after)


def test_no_movie_regression_even_with_pooled_gain():
    before=metrics();before['per_movie']['44b6_b']=dict(correct=1,total=5,annotated=7)
    before.update(correct=4,total=10)
    after=deepcopy(before);after.update(correct=5,ce=.8)
    after['per_movie']['44b6_a']['correct']=2;after['per_movie']['44b6_b']['correct']=3
    assert not checked_gate(before,after)
