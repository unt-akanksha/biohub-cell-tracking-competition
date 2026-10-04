import ast
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-joint-training.py'))


def function(source,name):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)


def test_original_live_forward_and_unweighted_diagnostic_preserved():
    source=BUILD['worker']();smoke=BUILD['SMOKE']['worker']();head=BUILD['SMOKE']['OLD']['BASE']['worker']()
    assert ast.dump(function(source,'forward'))==ast.dump(function(smoke,'forward'))
    assert ast.dump(function(source,'evaluate'))==ast.dump(function(head,'evaluate'))
    assert "require_fitting_sample(packets['fitting'][queue.pop()])" in source
    assert 'joint_update(model,optimizer,scaler,sample,forward' in source
    assert 'budget_checkpointed_incomplete_joint_training' in source
    assert 'remaining_sampling_queue=queue.copy()' in source and 'scaler=scaler.state_dict()' in source
    assert source.index('Initial diagnostic replay failed')<source.index('optimizer=torch.optim.AdamW')


def test_offline_budget_and_inputs():
    nb,meta=BUILD['build']({'contract':{}})
    assert len(meta['kernel_sources'])==3 and meta['enable_gpu'] and not meta['enable_internet']
    assert nb['metadata']['codex']['declared_budget_seconds']==1800
    launch=''.join(nb['cells'][-1]['source'])
    assert '1680-' in launch and 'declared_budget_seconds=1800' in launch and '3480-' not in launch
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
