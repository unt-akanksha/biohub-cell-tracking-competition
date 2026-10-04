import ast
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_fixed_image_profile_rejects_unverified_larger_run():
    m=runpy.run_path(str(ROOT/'scripts/run-backward-flow-probe.py'))
    assert m['loss_profile']('sparse',1000)==dict(image_weight=0.,smoothness_weight=0.)
    assert m['loss_profile']('image',100)==dict(image_weight=.25,smoothness_weight=.01)
    for name,steps in [('image',1000),('unknown',100)]:
        with pytest.raises(ValueError): m['loss_profile'](name,steps)


def test_probe_offline_bounded_and_embeds_tested_image_loss():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-backward-flow-image-probe.py'))['build']()
    assert meta['enable_internet'] is False and not meta['kernel_sources']
    assert nb['metadata']['codex']['max_steps']==100
    assert nb['metadata']['codex']['authorized_for_submission'] is False
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    bundle=ast.literal_eval(assignment.value)
    assert bundle['backward_flow_image_loss.py']==(ROOT/'research/backward_flow_image_loss.py').read_text()
    assert 'tests/test_backward_flow_image_loss.py' in bundle
    assert "command.extend(['--loss-profile','image'])" in ''.join(nb['cells'][-1]['source'])
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
