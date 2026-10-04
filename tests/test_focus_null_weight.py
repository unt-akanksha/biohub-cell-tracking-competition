import math
import pytest
from research.focus_null_balanced_loss import fitting_null_weight


def test_weight_comes_from_fixed_fitting_counts_only():
    assert fitting_null_weight(10754,161)==math.sqrt(10754/161)


@pytest.mark.parametrize('parent,absent',[(0,1),(1,0),(-1,1),(True,1),(1,1.5)])
def test_invalid_counts_rejected(parent,absent):
    with pytest.raises(ValueError):fitting_null_weight(parent,absent)
