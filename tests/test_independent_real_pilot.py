import ast
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_offline_pilot_contains_fresh_sources_without_model_materialization():
    module = runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))
    nb, meta = module['build']()
    sources = [''.join(c['source']) for c in nb['cells']]
    for s in sources:
        ast.parse(s)
    assert not meta['enable_internet'] and meta['enable_gpu'] and not meta['enable_tpu']
    assert len(meta['dataset_sources']) == 1 and not meta['kernel_sources']
    assert not any('materialize_inference_repo(' in s for s in sources)
    assert '3480-' in sources[-1] and 'signal.SIGKILL' in sources[-1]
    runner = (ROOT/'scripts/run-independent-real-pilot.py').read_text()
    assert 'torch.load(' not in runner
    assert "fold['train'][:4]" in runner
    assert 'GradScaler' in runner and 'os.replace' in runner
    assert "max_iters=10" in runner
    assert "math.log(.01/.99)" in runner
    assert "range(1, 51)" in runner
    assert runner.index("event='detector_warmup'") < runner.index('for step in range(10, args.steps+1, 10)')
    assert runner.index('if step > 100') < runner.index('edge, detection = epoch')
    assert "args.output/'smoke_checkpoint.pt'" in runner
    assert 'real_checkpoint_gpu_smoke.py' in sources[1]


def test_fixed_optimization_profile_does_not_expand_wall_budget():
    module = runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))
    nb, _ = module['build'](1000)
    assert nb['metadata']['codex']['max_steps'] == 1000
    assert nb['metadata']['codex']['declared_budget_seconds'] == 3600
    assert "'--steps', '1000'" in ''.join(nb['cells'][-1]['source'])
