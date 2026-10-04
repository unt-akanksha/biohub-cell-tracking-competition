import ast
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke.py'))


def function(source,name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)


def test_physics_evaluation_checkpoint_unchanged():
    actual=BUILD['worker']();original=(ROOT/'scripts/train-focus-adaptation-head.py').read_text()
    for name in ('forward','evaluate','save'):
        assert ast.dump(function(actual,name))==ast.dump(function(original,name))
    assert 'SETTINGS=dict(SETTINGS,steps=4)' in actual
    assert 'final=None;checkpoint=smoke' in actual
    assert actual.index('Complete audited fitting augmentations required')<actual.index('optimizer=torch.optim.AdamW')
    assert "sample=require_fitting_sample(smoke_samples[step-1])" in actual
    assert "loss=null_balanced_parent_loss(scores,labels,null_weight)" in actual


def test_private_offline_and_fifteen_minute_cap():
    nb,meta=BUILD['build']({})
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert len(meta['kernel_sources'])==3
    assert nb['metadata']['codex']['declared_budget_seconds']==900
    assert '780-' in ''.join(nb['cells'][-1]['source'])
    assert 'threading.Timer(840,' in ''.join(nb['cells'][0]['source'])
