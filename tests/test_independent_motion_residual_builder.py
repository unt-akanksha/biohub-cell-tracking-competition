import ast
from pathlib import Path
import runpy


def test_small_gpu_run_requires_numerical_gate_and_exact_initialization():
    root = Path(__file__).resolve().parents[1]
    nb,meta = runpy.run_path(str(root/'scripts/build-independent-motion-residual.py'))['build']()
    launch = ''.join(nb['cells'][-1]['source'])
    assert "'tests/test_motion_residual.py'" in launch
    assert "command.append('--motion-residual')" in launch
    assert launch.index('if gate.returncode != 0') < launch.index('process = subprocess.Popen')
    assert "'--steps','100'" in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-real-pilot-v1/4']
    assert not meta['enable_internet']
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))


def test_residual_selection_mount_and_hash_are_bound_to_trained_kernel():
    root = Path(__file__).resolve().parents[1]
    nb,meta = runpy.run_path(str(root/'scripts/build-independent-motion-residual-selection.py'))['build']('1'*64,1)
    launch = ''.join(nb['cells'][-1]['source'])
    assert 'independent_motion_residual/outputs/last.pt' in launch
    assert "'notebooks/indarkarhana/biohub-independent-motion-residual-v1'" in launch
    assert '1'*64 in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-motion-residual-v1/1']
    assert nb['metadata']['codex']['run_id'] == 'independent-motion-residual-selection-v1'


def test_row_negative_profile_has_separate_outputs_and_both_numerical_gates():
    root = Path(__file__).resolve().parents[1]
    nb,meta = runpy.run_path(str(root/'scripts/build-independent-motion-residual.py'))['build'](True)
    launch = ''.join(nb['cells'][-1]['source'])
    assert meta['id'] == 'indarkarhana/biohub-independent-motion-row-v1'
    assert 'tests/test_sparse_parent_row_loss.py' in launch
    assert "command.append('--row-negatives')" in launch
    assert '/kaggle/working/independent_motion_row' in ''.join(nb['cells'][0]['source'])
    assert nb['metadata']['codex']['max_steps'] == 100
