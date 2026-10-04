import ast
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/run-owned-detector-pu-probe.py'))


def test_larger_training_needs_exact_verified_raw_probe():
    raw=(ROOT/'.biohub/cache/kernel-outputs/owned-detector-logit-probe-v1/owned_detector_logit_probe/outputs/result.json').read_bytes()
    assert M['training_profile']('probe','pu',False)==(10,None)
    for objective in ('sparse','pu'):
        steps,probe=M['training_profile']('full',objective,True,raw)
        assert steps==1000 and probe['steps']==10
    for profile,objective,logits,payload in [('full','pu',False,raw),('full','pu',True,raw+b' '),
        ('full','invalid',True,raw),('probe','sparse',True,None)]:
        with pytest.raises(ValueError): M['training_profile'](profile,objective,logits,payload)


def test_builder_preserves_probes_and_launches_serial_pair_with_watchdog():
    original=ROOT/'kaggle/biohub-owned-detector-logit-probe-v1/biohub-owned-detector-logit-probe-v1.ipynb'
    before=original.read_bytes()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-owned-detector-fit-pair.py'))['build']()
    assert original.read_bytes()==before
    assert meta['id']=='indarkarhana/biohub-owned-detector-fit-pair-v1' and not meta['enable_internet']
    assert nb['metadata']['codex']['max_steps']==1000 and nb['metadata']['codex']['arms']==['sparse','pu']
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    assert M['training_profile']('full','pu',True,runtime['raw_probe_result.json'].encode())[0]==1000
    assert "for arm in ('sparse','pu')" in runtime['run_pair.py']
    assert 'subprocess.run(' in runtime['run_pair.py'] and 'start_new_session' not in runtime['run_pair.py']
    assert 'step%50==0: save(step)' in runtime['run_pilot.py']
    assert 'first10_graph_gate_passed=True' in runtime['run_pilot.py']
    assert "str(runtime/'run_pair.py')" in ''.join(nb['cells'][-1]['source'])
    assert "command.append('--logit-targets')" not in ''.join(nb['cells'][-1]['source'])
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
