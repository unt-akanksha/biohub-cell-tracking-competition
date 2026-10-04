import ast
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-joint-backoff-smoke.py'))


def test_backoff_preserves_forward_replay_and_only_replaces_update():
    source=BUILD['worker']();old=BUILD['OLD']['worker']()
    def forward(s):return next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert ast.dump(forward(source))==ast.dump(forward(old))
    assert 'joint_update(model,optimizer,scaler,sample,forward' in source
    assert 'all_four_original_image_replays_passed' in source
    assert 'Exact real-image checkpoint reload failed' in source
    helper=(ROOT/'research/focus_joint_probe_backoff.py').read_text()
    assert 'error_if_nonfinite=True' in helper and "max_amp_attempts=17" in helper
    assert 'torch.set_rng_state(cpu_rng)' in helper and 'parameters_unchanged=True' in helper
    assert 'tensor_hash(model.state_dict())!=before' in helper


def test_offline_inputs_and_declared_cap_unchanged():
    nb,meta=BUILD['build']({'contract':{}})
    assert meta['enable_gpu'] and not meta['enable_internet'] and len(meta['kernel_sources'])==3
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
