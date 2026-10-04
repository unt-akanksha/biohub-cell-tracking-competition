from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/verify-focus-pair-appearance-conditioned.py'))['check_optimizer_state']


def values():
    theta = np.zeros(72).tolist()
    model = dict(theta=theta, iterations=30, evaluations=37, objective=1.)
    state = dict(model, beta=theta.copy(), success=True)
    return state, np.eye(72), model


def test_actual_optimizer_coordinates_replay():
    CHECK(*values())


@pytest.mark.parametrize('mutation', ['failed', 'theta', 'beta', 'objective', 'bounds'])
def test_rejects_corrupt_or_failed_optimizer_state(mutation):
    state, transform, model = values()
    state = deepcopy(state)
    if mutation == 'failed':
        state['success'] = False
    elif mutation == 'theta':
        state['theta'][0] = .1
    elif mutation == 'beta':
        state['beta'][0] = .1
    elif mutation == 'objective':
        state['objective'] = 2.
    else:
        state['theta'][4] = .6
        state['beta'][4] = .6
        model['theta'][4] = .6
    with pytest.raises((ValueError, AssertionError)):
        CHECK(state, transform, model)
