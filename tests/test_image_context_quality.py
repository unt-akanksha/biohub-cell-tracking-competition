import numpy as np
from research.image_context_quality import metrics


def test_tied_false_positive_cannot_be_excluded_from_threshold():
    row = metrics([1, 1, 0], [3., 2., 2.])
    assert row['zero_fp_tp'] == 1 and row['source_threshold'] is None
    assert not row['source_gate_passed']
    assert abs(row['average_precision'] - 5/6) < 1e-12


def test_source_threshold_is_not_reselected_on_target():
    source = metrics([1, 1, 0, 0], [4., 3., 2., 1.])
    assert source['source_threshold'] == 3.
    target = metrics([1, 0, 1, 0], [2., 4., 3.5, 0.], threshold=source['source_threshold'])
    assert target['fixed_threshold'] == dict(threshold=3., tp=1, fp=1, fn=1, jaccard=1/3)


def test_all_ties_ap_is_prevalence_not_input_order():
    for labels in ([1, 1, 0, 0], [0, 1, 0, 1]):
        assert metrics(labels, np.zeros(4))['average_precision'] == .5
