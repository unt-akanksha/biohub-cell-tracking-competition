from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/diagnose-frozen-detector-selection.py'))


def test_exhaustive_missing_edge_attribution_with_sparse_unknown_predictions():
    truth=[(1,2),(3,4),(5,6),(7,8),(9,10)]
    mapping={101:1,102:2,103:4,104:5,105:9,106:10,107:-1}
    result=M['decompose_edges'](truth,mapping,[(101,102),(104,107)])
    assert result['recovered']==1
    assert result['missing_both_endpoints']==1
    assert result['missing_source_only']==1
    assert result['missing_target_only']==1
    assert result['both_endpoints_detected_but_not_linked']==1
    assert result['unrecoverable_without_detection_change']==3
    assert result['association_recall_given_both_detected']==.5


def test_duplicate_prediction_matches_do_not_inflate_recovered_gt_edges():
    result=M['decompose_edges']([(1,2)],{10:1,11:1,12:2},[(10,12),(11,12),(10,12)])
    assert result['recovered']==1 and result['total_gt_edges']==1


def test_duplicate_truth_rejected_and_empty_opportunities_are_not_a_score():
    with pytest.raises(ValueError): M['decompose_edges']([(1,2),(1,2)],{},[])
    assert M['decompose_edges']([(1,2)],{},[])['association_recall_given_both_detected'] is None
