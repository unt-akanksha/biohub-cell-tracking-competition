import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-gt-parent-dropout-smoke.py'))


def function(source,name):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)


def test_only_fitting_augmentation_changes_not_physics_or_diagnostics():
    source=BUILD['worker']();old=(ROOT/'scripts/train-focus-adaptation-head.py').read_text()
    for name in ('forward','evaluate','save'):assert ast.dump(function(source,name))==ast.dump(function(old,name))
    assert "altered=augment(sample,row['parent_gt_coords'])" in source
    assert 'SETTINGS=dict(SETTINGS,steps=4)' in source
    assert 'final=None;checkpoint=smoke' in source
    assert source.index('Complete audited fitting augmentations required')<source.index('optimizer=torch.optim.AdamW')


def test_exact_emitted_metadata_offline_and_budget():
    bundle={'records':[],'marker':'\r\n'};nb,meta=BUILD['build']({},bundle)
    runtime=BUILD['decode_runtime'](''.join(nb['cells'][1]['source']))
    assert json.loads(runtime['dropout_audit.json'])==bundle
    assert 'focus_gt_parent_dropout.py' in runtime
    assert meta['enable_gpu'] and meta['is_private'] and not meta['enable_internet']
    assert nb['metadata']['codex']['declared_budget_seconds']==900
    assert '780-' in ''.join(nb['cells'][-1]['source'])
