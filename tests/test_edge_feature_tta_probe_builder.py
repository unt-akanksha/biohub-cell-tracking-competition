import ast
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_probe_is_hash_bound_training_only_and_pretested():
    module = runpy.run_path(str(ROOT/'scripts/build-edge-feature-tta-probe.py'))
    nb,meta = module['build']()
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-broad-v1/1']
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert nb['metadata']['codex']['declared_budget_seconds'] == 3600
    launch = ''.join(nb['cells'][-1]['source'])
    assert module['CHECKPOINT_SHA'] in launch
    assert "runtime/'run_feature_probe.py'" in launch
    assert launch.index('test_edge_feature_tta.py') < launch.index('process = subprocess.Popen')
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    assert runtime['edge_feature_tta.py'] == (ROOT/'research/edge_feature_tta.py').read_text()
    assert 'native != augmented' in runtime['run_feature_probe.py']
    assert "stem = fold['train'][0]" in runtime['run_feature_probe.py']
    assert 'encode_patch=None' in runtime['real_checkpoint_gpu_smoke.py']
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
