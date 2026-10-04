import ast
from pathlib import Path
import runpy


def test_frozen_loader_no_optimizer_and_small_replay_first():
    root=Path(__file__).resolve().parents[1]
    code=runpy.run_path(str(root/'scripts/build-focus-presence-summary.py'))['worker']()
    ast.parse(code)
    assert 'optimizer.step' not in code and 'AdamW' not in code
    assert 'model.requires_grad_(False).eval()' in code
    assert code.index('previous_diagnostic_replayed')<code.index('for role,samples in packets.items()')
    assert 'Presence factorization differs' in code and 'Frozen head changed' in code
