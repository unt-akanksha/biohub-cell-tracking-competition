import ast
from pathlib import Path
import pytest
from research.independent_real_baseline import make_split, fresh_batches, patch_train_epoch


def test_embryo_audit_never_enters_training_or_selection():
    stems = [f'{e}_{i:08x}' for e in ('44b6', '6bba') for i in range(20)]
    result = make_split(stems)
    assert result == make_split(stems[::-1])
    for fold in result['folds']:
        assert set(fold['train']).isdisjoint(fold['selection'])
        assert set(fold['train'] + fold['selection']).isdisjoint(fold['audit_order'])
        assert all(s.startswith(fold['held_out_embryo']) for s in fold['audit_order'])
        assert len(fold['initial_complete_movie_audit']) == 4


def test_invalid_inventory_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        make_split(['44b6_a', '44b6_a'])
    with pytest.raises(ValueError, match='Unexpected'):
        make_split(['test_a'])


def test_batch_repetition_reopens_loader_instead_of_caching_tensors():
    class Loader:
        def __init__(self):
            self.passes = 0
        def __iter__(self):
            self.passes += 1
            yield self.passes
    loader = Loader()
    batches = fresh_batches(loader)
    assert [next(batches) for _ in range(3)] == [1, 2, 3]
    with pytest.raises(ValueError, match='Empty'):
        next(fresh_batches([]))


def test_amp_patch_matches_actual_pinned_organizer_source():
    root = Path(__file__).resolve().parents[1]
    source = (root / '.biohub/vendor/kaggle-cell-tracking-competition/scripts/train_unet_transformer.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
    function = ast.get_source_segment(source, node)
    patched = patch_train_epoch(function)
    assert 'scaler.unscale_(optimizer)' in patched
    assert patched.index('scaler.unscale_') < patched.index('clip_grad_norm_')
    assert '_cycle(loader)' not in patched
    with pytest.raises(ValueError, match='source drift'):
        patch_train_epoch(patched)
