import ast
from pathlib import Path
import runpy


def test_head_only_offline_bounded_runtime():
    root=Path(__file__).resolve().parents[1]
    nb,meta=runpy.run_path(str(root/'scripts/build-focus-adaptation-head.py'))['build']({'contract':{}})
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu'] and meta['is_private']
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-focus-adaptation-features-v1/1']
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    last=''.join(nb['cells'][-1]['source'])
    assert "'--features-root', str(features_root)" in last and '3480-' in last and 'SIGTERM' in last
    assert "'pytest'" not in last
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    for name,source in runtime.items():
        if name.endswith('.py'):ast.parse(source)
    worker=runtime['run_pilot.py']
    assert 'model.requires_grad_(False)' in worker and 'AdamW(model.transformer.parameters()' in worker
    assert "require_fitting_sample(packets['fitting']" in worker
    assert 'weights_only=True' in worker and 'Small real training probe has no gradient' in worker
