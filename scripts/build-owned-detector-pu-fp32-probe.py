"""Separate repaired numerical probe; preserve the original pilot notebook."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-owned-detector-pu-fp32-probe-v1'


def build():
    gate=json.loads((ROOT/'.biohub/cache/kernel-outputs/owned-detector-pu-smoke-v2/owned_detector_pu_smoke/test_result.json').read_text())
    if gate['exit_code']!=0 or not gate['cpu_only'] or '11 passed' not in gate['stdout']:
        raise ValueError('FP32 probability-order CPU gate required')
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-owned-detector-pu-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['run_pilot.py']=runtime['run_pilot.py'].replace('owned-detector-pu-probe-v1','owned-detector-pu-fp32-probe-v1')
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('owned-detector-pu-probe-v1','owned-detector-pu-fp32-probe-v1')
        source=source.replace('owned_detector_pu_probe','owned_detector_pu_fp32_probe')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='owned-detector-pu-fp32-probe-v1',teacher_probability_precision='float32_before_sigmoid')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector PU FP32 Probe v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
