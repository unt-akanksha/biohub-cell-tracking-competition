import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from research.trajectory_event_fork16_worker_v1 import adapt_worker, NEW_CALL, OLD_CALL, FORK_GUARD

ROOT = Path(__file__).resolve().parents[1]


def original():
    return json.loads((ROOT / '.biohub/staging/biohub-event-anchor-candidate-v2/overlays.json').read_text())['portable-worker.py']


def test_only_vocabulary_guard_and_explicit_deadlines_change():
    source = original()
    adapted = adapt_worker(source)
    assert adapted.replace(NEW_CALL, OLD_CALL).replace(FORK_GUARD, '') == source
    assert adapted.count(NEW_CALL) == 1
    ast.parse(adapted)


def test_actual_adapted_call_passes_frozen_deadlines():
    tree = ast.parse(adapt_worker(original()))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'event' and n.func.attr == 'refine']
    assert len(calls) == 1
    captured = {}
    def spy(*args, **kwargs):
        captured.update(kwargs)
        return args
    env = dict(event=SimpleNamespace(refine=spy), raw_ilp={}, repaired={}, coords=np.zeros((0, 4)),
               np=np, edges=[], event_model=dict(weights=np.zeros(30)))
    eval(compile(ast.Expression(calls[0]), '<adapted-event-call>', 'eval'), env)
    assert captured == dict(per_frame_seconds=10., max_seconds=600.)


@pytest.mark.parametrize('value', [None, 8, 0])
def test_incompatible_vocabulary_rejected(value):
    source = 'def check(event_model):\n' + FORK_GUARD
    env = {}
    exec(source, env)
    with pytest.raises(ValueError, match='fork16'):
        env['check'](dict(max_fork_children=value))


def test_double_adaptation_rejected():
    with pytest.raises((ValueError, AssertionError)):
        adapt_worker(adapt_worker(original()))
