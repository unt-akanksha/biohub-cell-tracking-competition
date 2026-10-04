import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]


def test_offline_frozen_inputs_and_watchdog():
    builder=runpy.run_path(str(ROOT/'scripts/build-focus-adaptation-features.py'))
    nb,meta=builder['build']({'contract':{},'movies':[]})
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu'] and meta['is_private']
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-focus-adaptation-cache-v1/1','indarkarhana/biohub-focus-owned-neural-probe-v1/1']
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    last=''.join(nb['cells'][-1]['source'])
    assert "'--raw-root', str(raw_root)" in last and "'--probe-root', str(probe_root)" in last
    assert '3480-' in last and 'SIGTERM' in last and "'pytest'" not in last
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    for name,value in runtime.items():
        if name.endswith('.py'):ast.parse(value)
    assert 'validate_spec(spec,scope' in runtime['run_pilot.py']
    assert 'Both prior GPU replays must pass' in runtime['run_pilot.py']
    assert 'optimizer' not in runtime or not runtime['optimizer']
    assert json.loads(runtime['features_spec.json'])=={'contract':{},'movies':[]}
