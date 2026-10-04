import ast
from pathlib import Path
import runpy


def test_fixed_cache_verifier_reuses_full_eight_movie_guards():
    root = Path(__file__).resolve().parents[1]
    scope = runpy.run_path(str(root / 'scripts/verify-trajectory-overlap-cache-v1.py'))
    source = scope['configured_source']()
    ast.parse(source)
    assert 'args.overlap=True' in source
    assert '99c9fb48404e31051ce3096a0901b32698d19742abb3f04d5f4dc0453f9ec411' in source
    assert "if set(prepared)!=set(STEMS):raise ValueError('All eight movies required before truth access')" in source
    reject = source.index("raise ValueError('Exact-cache runtime changed a graph; reject without rescoring')")
    assert reject < source.index('    ground_truth_opened=False')
    assert 'trajectory-overlap-cache-kaggle-v1-result.json' in source
