import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-training.py'))


def function(source,name):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)


def test_unchanged_physics_and_complete_diagnostic():
    source=BUILD['worker']();old=(ROOT/'scripts/train-focus-adaptation-head.py').read_text()
    for name in ('forward','evaluate'):assert ast.dump(function(source,name))==ast.dump(function(old,name))
    assert 'SETTINGS=dict(SETTINGS,steps=800)' in source
    assert "stream='original' if step%2 else 'augmented'" in source
    assert 'remaining_sampling_queues={k:v.copy() for k,v in queues.items()}' in source
    assert 'final=evaluate()' in source and 'gate=diagnostic_gate(initial,physical,final)' in source
    assert source.index('Initial diagnostic replay failed')<source.index('optimizer=torch.optim.AdamW')


def test_exact_audit_bytes_and_private_bounded_runtime():
    nb,meta=BUILD['build']({});runtime=BUILD['decode_runtime'](''.join(nb['cells'][1]['source']))
    assert runtime['dropout_audit.json'].encode()==(ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json').read_bytes()
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert nb['metadata']['codex']['declared_budget_seconds']==900
    assert '780-' in ''.join(nb['cells'][-1]['source'])
    assert len(json.dumps(nb).encode())<950000
