"""Offline ten-update real detector functionality gate after CPU PU tests."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-owned-detector-pu-probe-v1'


def build():
    gate=json.loads((ROOT/'.biohub/cache/kernel-outputs/owned-detector-pu-smoke-v1/owned_detector_pu_smoke/test_result.json').read_text())
    if gate['exit_code']!=0 or not gate['cpu_only'] or not any(s in gate['stdout'] for s in ('10 passed','11 passed')):
        raise ValueError('Real CPU target/gradient gate required')
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['run_pilot.py']=(ROOT/'scripts/run-owned-detector-pu-probe.py').read_text()
    for name in ('owned_detector_pu','seeded_frame_dataset'):
        runtime[name+'.py']=(ROOT/f'research/{name}.py').read_text()
    runtime['spotiflow_biohub/pu_targets.py']=(ROOT/'research/spotiflow_biohub/pu_targets.py').read_text()
    for name in ('owned_detector_pu','owned_detector_pu_gradients'):
        runtime[f'tests/test_{name}.py']=(ROOT/f'tests/test_{name}.py').read_text().replace(
            "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-probe-v1','owned-detector-pu-probe-v1')
        source=source.replace('detector_spatial_tta_probe','owned_detector_pu_probe')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='owned-detector-pu-probe-v1',max_steps=10,
        scope='Ten updates on four original source-training movies; frozen owned teacher, linker, flow and BN statistics')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector PU Probe v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
