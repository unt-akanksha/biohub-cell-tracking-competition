import ast
import copy
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('steps', [100, 6000])
def test_longer_profile_is_bounded_and_pretested(steps):
    builder = runpy.run_path(str(ROOT / 'scripts/build-independent-joint-extended.py'))
    nb, meta = builder['build'](steps)
    launch = ''.join(nb['cells'][-1]['source'])
    assert f"'--steps','{steps}'" in launch
    assert "'--joint','--checkpoint-bn-once'" in launch
    assert builder['CHECKPOINT_SHA'] in launch
    assert 'independent_joint_broad/outputs/last.pt' in launch
    assert 'independent_motion_broad_pair/outputs/control' not in launch
    assert launch.index("str(runtime/'tests/test_checkpoint_bn_guard.py')") < launch.index('process = subprocess.Popen')
    assert '7080-(time.monotonic()-started)' in launch
    assert 'threading.Timer(7140,' in ''.join(nb['cells'][0]['source'])
    assert nb['metadata']['codex']['declared_budget_seconds'] == 7200
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-broad-v1/1']
    assert meta['enable_gpu'] and not meta['enable_internet']
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    assert runtime['checkpoint_bn_guard.py'] == (ROOT / 'research/checkpoint_bn_guard.py').read_text()
    runner = runtime['run_association.py']
    assert "run_id='independent-joint-extended-v1'" in runner
    assert "if step > 100 and (step100_smoke is None" in runner
    assert "value != step for value in deltas.values()" in runner
    assert "checkpoint_bn_once=True" in runner
    assert 'scaler.state_dict()' in runner
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))


def test_unplanned_training_length_rejected():
    builder = runpy.run_path(str(ROOT / 'scripts/build-independent-joint-extended.py'))
    with pytest.raises(ValueError):
        builder['build'](10000)


def test_short_or_skipped_optimizer_fit_cannot_enter_selection():
    check = runpy.run_path(str(ROOT / 'scripts/run-independent-selection-inference.py'))['check_completed_profile']
    state = dict(step=6000, identity=dict(max_steps=6000, checkpoint_bn_once=True,
        joint_training=True, training_profile='full', motion_residual={'version':1},
        loss='sparse_parent_classification_with_null_v1'), optimizer=dict(state={0:dict(step=6000)}))
    check(state)
    for steps in (100, 1000, 5990):
        bad = copy.deepcopy(state)
        bad['step'] = steps
        bad['identity']['max_steps'] = steps
        with pytest.raises(ValueError):
            check(bad)
    state['optimizer']['state'][0]['step'] = 5300
    with pytest.raises(ValueError, match='optimizer'):
        check(state)


def test_extended_scoring_embeds_exact_selection_notebook(monkeypatch):
    nb, meta = runpy.run_path(str(ROOT / 'scripts/build-independent-extended-selection.py'))['build']('a'*64, 2)
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-extended-v1/2']
    assert 'independent_joint_extended/outputs/last.pt' in ''.join(nb['cells'][-1]['source'])
    frozen = json.dumps(nb)
    path = ROOT / 'kaggle/biohub-independent-extended-selection-v1/biohub-independent-extended-selection-v1.ipynb'
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda self,*a,**k: frozen if self == path else original(self,*a,**k))
    scored, meta = runpy.run_path(str(ROOT / 'scripts/build-independent-extended-scoring.py'))['build'](1)
    source = ''.join(scored['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'scoring_sources')
    assert ast.literal_eval(assignment.value)['selection_launch.ipynb'] == frozen
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-extended-selection-v1/1']
    assert not meta['enable_gpu']
