import ast
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]


def test_real_probe_is_paired_training_only_with_fixed_motion():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-probe.py'))['build']()
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2']
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    assert runtime['run_pilot.py'].count('standalone_image_flow=True')==2
    assert 'pre_motion_patch=install_detector_spatial_tta' in runtime['run_pilot.py']
    assert "stem=fold['train'][0]" in runtime['run_pilot.py']
    assert 'tests/test_detector_spatial_tta.py' in runtime
    assert 'tests/test_image_motion_residual.py' in runtime
    smoke=ast.parse(runtime['real_checkpoint_gpu_smoke.py'])
    text=runtime['real_checkpoint_gpu_smoke.py']
    assert text.index('pre_motion_patch(model)') < text.index('install_image_motion_residual(model,')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
