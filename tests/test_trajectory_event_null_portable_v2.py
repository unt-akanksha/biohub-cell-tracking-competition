import ast
from pathlib import Path
import runpy

import numpy as np

from research.trajectory_event_null_dominance_v1 import allowed_options

ROOT = Path(__file__).resolve().parents[1]
builder = runpy.run_path(str(ROOT / 'scripts/build-trajectory-event-null-portable-v2.py'))


def test_standalone_only_intended_function_replacement():
    old = ast.parse((ROOT / '.biohub/cache/trajectory-event-fork16-vectorized-portable-v1/event-trajectory.py').read_text())
    new = ast.parse(builder['portable_source']())
    original_functions = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n) for n in new.body if isinstance(n, ast.FunctionDef)}
    for name in original_functions.keys() - {'allowed_options'}:
        assert original_functions[name] == new_functions[name]
    renamed = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'fork_pruning')
    renamed.name = 'allowed_options'
    assert ast.dump(renamed) == original_functions['allowed_options']
    assert not any(isinstance(n, ast.ImportFrom) and (n.module or '').startswith('research') for n in ast.walk(new))
    assert set(new_functions) == set(original_functions) | {'fork_pruning'}


def test_standalone_pruner_matches_research_on_masks_and_scores():
    namespace = {'__name__': 'standalone_test'}
    exec(compile(builder['portable_source'](), 'standalone_test', 'exec'), namespace)
    options = np.array([[-1, 0, -1], [-1, 1, -1], [0, -1, -1],
                        [0, 0, -1], [0, 1, -1], [0, 0, 1]])
    case = dict(options=options, nparents=1, nchildren=2)
    rng = np.random.default_rng(20260914)
    for _ in range(100):
        scores = rng.normal(size=len(options))
        mask = rng.random(len(options)) > .2
        expected, expected_report = allowed_options(case, scores, mask)
        actual, report = namespace['allowed_options'](case, scores, mask)
        np.testing.assert_array_equal(actual, expected)
        assert report == expected_report
