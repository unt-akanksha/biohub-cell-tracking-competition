import ast
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_association_pilot_binds_initialization_and_preserves_small_scope():
    builder = runpy.run_path(str(ROOT/'scripts/build-independent-association-pilot.py'))
    nb,meta = builder['build']()
    launch = ''.join(nb['cells'][-1]['source'])
    assert "runtime/'run_association.py'" in launch
    assert "'notebooks/indarkarhana/biohub-independent-real-pilot-v1'" in launch
    assert builder['CHECKPOINT_SHA'] in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-real-pilot-v1/4']
    assert not meta['enable_internet'] and nb['metadata']['codex']['max_steps'] == 100
    source = (ROOT/'scripts/run-independent-association-pilot.py').read_text()
    ast.parse(source)
    assert 'range(10,args.steps+1,10)' in source
    assert 'module.requires_grad_(False)' in source
    assert "if not args.joint and detector_hash() != frozen_hash:" in source
    assert "before = run_smoke" in source and "after = run_smoke" in source


def test_extension_is_step_bound_and_gpu_smoke_gated():
    builder = runpy.run_path(str(ROOT/'scripts/build-independent-association-pilot.py'))
    nb,_ = builder['build'](1000)
    assert nb['metadata']['codex']['max_steps'] == 1000
    assert "'--steps','1000'" in ''.join(nb['cells'][-1]['source'])
    source = (ROOT/'scripts/run-independent-association-pilot.py').read_text()
    assert "if step > 100 and (step100_smoke is None" in source
    assert "args.output/'step100.pt'" in source
