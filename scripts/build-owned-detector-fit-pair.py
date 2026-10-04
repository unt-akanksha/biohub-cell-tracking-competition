"""Paired source120 detector fits after verified ten-step raw-target probe."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-owned-detector-fit-pair-v1'


def build():
    report=json.loads((ROOT/'reports/experiments/owned-detector-logit-probe-v1-result.json').read_text())
    if (report['status']!='verified_owned_detector_pu_functionality_not_selection' or not report['all_frozen_components_unchanged']
        or report['checkpoint_sha256']!='f7d840286d1c7d1ae1336efbb595fca0c9d328f7b11ccb7cd2071c10a4280aa9'):
        raise ValueError('Verified real raw-target probe required')
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-owned-detector-logit-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['run_pilot.py']=runtime['run_pilot.py'].replace('owned-detector-logit-probe-v1','owned-detector-fit-pair-v1')
    runtime['run_pair.py']=(ROOT/'scripts/run-owned-detector-fit-pair.py').read_text()
    runtime['raw_probe_result.json']=(ROOT/'.biohub/cache/kernel-outputs/owned-detector-logit-probe-v1/owned_detector_logit_probe/outputs/result.json').read_bytes().decode()
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('owned-detector-logit-probe-v1','owned-detector-fit-pair-v1')
        source=source.replace('owned_detector_logit_probe','owned_detector_fit_pair')
        if index==len(nb['cells'])-1:
            source=source.replace("str(runtime/'run_pilot.py')","str(runtime/'run_pair.py')")
            source=source.replace("command.append('--logit-targets')\n",'')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='owned-detector-fit-pair-v1',max_steps=1000,arms=['sparse','pu'],
        scope='Each arm: exact first10 probe replay, then990 updates over original120 source movies; sequential arms, frozen non-detector components')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector Fit Pair v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
