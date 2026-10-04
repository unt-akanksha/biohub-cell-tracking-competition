import ast
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-null-balanced-head.py'))


def function(source,name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)


def test_only_training_loss_changes_and_roles_stay_separate():
    source=BUILD['worker']();old=(ROOT/'scripts/train-focus-adaptation-head.py').read_text()
    for name in ('forward','evaluate','save'):
        assert ast.dump(function(source,name))==ast.dump(function(old,name))
    assert source.count('loss=null_balanced_parent_loss(scores,labels,null_weight)')==1
    assert "require_fitting_sample(packets['fitting'][queue.pop()])" in source
    assert "counts!=spec['supervision_counts']" in source
    assert source.index('Initial diagnostic replay failed')<source.index('optimizer=torch.optim.AdamW')
    assert "step==SETTINGS['smoke_steps']" in source
    assert 'New group cannot supply diagnostics' in source


def test_notebook_offline_pinned_and_bounded():
    nb,meta=BUILD['build']({'contract':{}})
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert len(meta['kernel_sources'])==3
    assert meta['kernel_sources'][-1]=='indarkarhana/biohub-focus-extra-fit-features-v1/1'
    final=''.join(nb['cells'][-1]['source'])
    assert '--extra-features-root' in final and '3480-' in final
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
