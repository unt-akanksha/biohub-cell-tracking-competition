import numpy as np
from research.trajectory_candidate_ranker_v1 import FEATURES, fit, evaluate, supervised, features
from research.trajectory_candidate_inventory_v1 import candidates


def test_fixed_regularized_fit_learns_a_predictive_feature():
    x = np.zeros((2,len(FEATURES))); x[1,0] = 2
    rows = [dict(x=x, y=0, parents=np.array([3,4]), child=5, current=4, initial=3)] * 10
    weights, receipt = fit(rows)
    report = evaluate(rows, weights)
    assert receipt['iterations'] > 0 and weights[0] < 0
    assert report['learned_correct'] == 10 and report['learned_repairs'] == 10


def test_unknown_and_ambiguous_candidates_never_enter_loss():
    groups = dict(children=np.array([5,6]), parents=np.array([1,2,3,1,2]),
                  offsets=np.array([0,3,5]), current=np.array([1,1]), neural=np.array([1,1]))
    rows = supervised(np.zeros((5,len(FEATURES))), groups, np.array([1,-1]),
                      np.array([True,False,True,False,False]))
    assert len(rows) == 1 and rows[0]['parents'].tolist() == [1,3]


def test_features_translation_invariant_and_missing_history_masked():
    import copy
    nodes = {0: dict(t=0,z=0.,y=0.,x=0.), 1: dict(t=0,z=0.,y=0.,x=30.),
             2: dict(t=1,z=0.,y=0.,x=1.), 3: dict(t=2,z=0.,y=0.,x=2.)}
    graph = dict(nodes=nodes, edges=[dict(source_id=0,target_id=2),dict(source_id=2,target_id=3)])
    groups = candidates(graph,graph)
    raw = np.array([[0,2,.9,1.],[2,3,.8,1.]])
    x = features(graph,graph,groups,raw)
    shifted = copy.deepcopy(graph)
    for n in shifted['nodes'].values():
        n['x'] += 100.; n['y'] += 20.; n['z'] += 10.
    assert np.array_equal(x,features(shifted,shifted,groups,raw))
    assert not x[:2,FEATURES.index('history_known')].any()
    assert not x[:2,FEATURES.index('full_history_residual')].any()
    assert 'current_edge' not in FEATURES
