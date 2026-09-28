"""One fixed, outcome-independent third-epoch checkpoint averaging rule."""
import numpy as np


def snapshot_steps(cases, epochs=3, period=50):
    if not isinstance(cases, int) or cases <= 0 or epochs != 3 or period != 50:
        raise ValueError('Expected three fixed epochs and the original 50-update snapshots')
    return list(range((2 * cases // period + 1) * period, 3 * cases + 1, period))


def average_snapshots(snapshots, cases):
    expected = snapshot_steps(cases)
    if not expected or set(snapshots) != set(expected):
        raise ValueError('Missing or extra third-epoch regular snapshot')
    weights = np.stack([np.asarray(snapshots[s], np.float64) for s in expected])
    if weights.shape != (len(expected), 30) or not np.isfinite(weights).all():
        raise ValueError('Invalid snapshot weight schema')
    return weights.mean(axis=0)
