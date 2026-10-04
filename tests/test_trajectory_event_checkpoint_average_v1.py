import numpy as np
import pytest
from research.trajectory_event_checkpoint_average_v1 import snapshot_steps, average_snapshots


def test_original_schedules_exclude_extra_epoch_end_checkpoints():
    a, b = snapshot_steps(1223), snapshot_steps(4557)
    assert len(a) == 25 and a[0] == 2450 and a[-1] == 3650
    assert len(b) == 91 and b[0] == 9150 and b[-1] == 13650
    assert 2446 not in a and 3669 not in a and 13671 not in b


def test_uniform_mean_and_order_independence():
    snapshots = {s: np.full(30, float(s)) for s in reversed(snapshot_steps(100))}
    expected = np.full(30, 275.)
    np.testing.assert_array_equal(average_snapshots(snapshots, 100), expected)


@pytest.mark.parametrize('kind', ['missing', 'extra', 'nonfinite', 'schema'])
def test_invalid_snapshot_inventory_fails(kind):
    rows = {s: np.zeros(30) for s in snapshot_steps(100)}
    if kind == 'missing': del rows[250]
    if kind == 'extra': rows[275] = np.zeros(30)
    if kind == 'nonfinite': rows[250][0] = np.nan
    if kind == 'schema': rows[250] = np.zeros(29)
    with pytest.raises(ValueError):
        average_snapshots(rows, 100)
