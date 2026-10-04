import numpy as np
import pytest
from research.trajectory_event_adam_state_v1 import initialize, update
from research import trajectory_event_fast_training_v1 as reference


def test_exact_legacy_optimizer_sequence(monkeypatch):
    cases = [dict(x=np.zeros((1, 3)), target=np.array([i, -i, .25])) for i in range(4)]
    def loss(case, weights, **kwargs):
        delta = weights - case['target']
        return float(delta @ delta / 2), delta
    monkeypatch.setattr(reference, 'hinge', loss)
    anchor, penalty = np.array([.2, -.1, .3]), np.array([.1, .1, .02])
    expected, _ = reference.fit(cases, anchor, penalty, epochs=3, seed=41)
    rng, state = np.random.default_rng(41), initialize(anchor)
    for epoch in range(3):
        for index in rng.permutation(len(cases)):
            _, gradient = loss(cases[index], state['weights'])
            update(state, gradient + penalty * (state['weights'] - anchor))
    assert np.array_equal(state['weights'], expected)
    assert state['steps'] == 12


def test_npz_roundtrip_resumes_identically(tmp_path):
    gradients = np.random.default_rng(3).normal(size=(12, 4)) * 10
    direct = initialize(np.zeros(4))
    for gradient in gradients:
        update(direct, gradient)
    partial = initialize(np.zeros(4))
    for gradient in gradients[:5]:
        update(partial, gradient)
    path = tmp_path / 'checkpoint.npz'
    np.savez_compressed(path, **partial)
    with np.load(path, allow_pickle=False) as data:
        restored = {k: data[k].copy() for k in ('weights', 'mean', 'variance')}
        restored['steps'] = int(data['steps'])
    for gradient in gradients[5:]:
        update(restored, gradient)
    for key in ('weights', 'mean', 'variance'):
        assert np.array_equal(restored[key], direct[key])
    assert restored['steps'] == direct['steps']


def test_rejected_gradient_does_not_mutate_state():
    state = initialize(np.zeros(3))
    before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
    with pytest.raises(ValueError):
        update(state, np.array([np.nan, 0, 1]))
    for key in ('weights', 'mean', 'variance'):
        assert np.array_equal(state[key], before[key])
    assert state['steps'] == 0
