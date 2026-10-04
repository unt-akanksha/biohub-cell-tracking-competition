import numpy as np
import pytest
from research.trajectory_correction_fp_labels_v2 import relabel


def record(index=0, delta=.01, **extra):
    return dict(index=index, delta_score=delta, old_label=dict(label=-1, known_children=1, **extra))


def test_positive_negative_and_neutral_retained_without_input_mutation():
    old = np.array([-1, -1, -1, 1, 0])
    result = relabel(old, [record(0), record(1, -.01), record(2, 1e-13)])
    assert result.tolist() == [1, 0, -1, 1, 0]
    assert old.tolist() == [-1, -1, -1, 1, 0]


@pytest.mark.parametrize('extra', [dict(unknown_children=1), dict(ambiguous_children=1)])
def test_mixed_unknown_rejected(extra):
    with pytest.raises(ValueError):
        relabel(np.array([-1]), [record(**extra)])


def test_existing_label_duplicate_and_nonfinite_rejected():
    for old, rows in [(np.array([1]), [record()]),
                      (np.array([-1]), [record(), record()]),
                      (np.array([-1]), [record(delta=float('nan'))])]:
        with pytest.raises(ValueError):
            relabel(old, rows)
