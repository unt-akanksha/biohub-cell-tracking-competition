import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_probe_is_bounded_offline_random_fit_with_runtime_gates():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-probe.py'))['build']()
    assert meta['kernel_sources'] == []
    assert meta['enable_gpu'] is True and meta['enable_internet'] is False
    assert nb['metadata']['codex']['max_steps'] == 100
    launch = ''.join(nb['cells'][-1]['source'])
    assert '--steps' not in launch
    assert "str(runtime/'tests')" in launch
    assert launch.index('subprocess.run') < launch.index('subprocess.Popen')
    assert '3480-' in launch
    bundle = next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body
        if isinstance(n,ast.Assign) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(bundle.value)
    assert 'backward_flow_model.py' in runtime and 'tests/test_backward_flow_model.py' in runtime
    worker = runtime['run_pilot.py']
    assert worker == (ROOT/'scripts/run-backward-flow-probe.py').read_text(encoding='utf-8')
    assert "fold['train'][::5]" in worker and 'len(fitting_stems) != 96' in worker
    assert 'min(after[\'zero_mae_um\'],after[\'median_mae_um\'])' in worker
    assert 'if after != reloaded:' in worker
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))


def test_cpu_smoke_is_cpu_only_and_runs_both_suites():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-model-smoke.py'))['build']()
    assert meta['enable_gpu'] is False and meta['enable_internet'] is False
    source = ''.join(nb['cells'][0]['source'])
    assert "str(work/'tests')" in source
    assert 'test_backward_flow_model.py' in source and 'test_backward_flow_ops.py' in source


def test_extension_keeps_probe_immutable_and_requires_replay():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-fit.py'))['build']()
    assert meta['kernel_sources'] == []
    assert nb['metadata']['codex']['max_steps'] == 1000
    assert nb['metadata']['codex']['step100_gate_required'] is True
    launch = ''.join(nb['cells'][-1]['source'])
    assert "'--steps','1000'" in launch
    assert 'backward-flow-fit-v1' in launch
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    assert json.loads(runtime['probe_result.json'])['small_fit_gate_passed'] is True
    assert "train.sample_hashes != probe['input_hashes']" in runtime['run_pilot.py']
    assert 'step100 motion-learning gate' in runtime['run_pilot.py'].lower()
