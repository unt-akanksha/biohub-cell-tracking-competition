"""Expand the verified frozen-network probe, with exact small-cache replay."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-association-calibration-fit-v1'


def build():
    report = json.loads((ROOT/'reports/experiments/association-calibration-probe-v1.json').read_text())
    probe_path = ROOT/'.biohub/cache/kernel-outputs/association-calibration-probe-v1/association_calibration_probe/outputs/result.json'
    probe = probe_path.read_text()
    if (report['status']!='verified_calibration_not_tracking_candidate' or report['profile']!='probe'
        or report['windows']!=18 or not report['result']['functionality_passed']
        or hashlib.sha256(probe_path.read_bytes()).hexdigest()!=report['source_sha256']['result']):
        raise ValueError('Verified immutable eighteen-pair probe required')
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-association-calibration-probe.py'))['build']()
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['probe_result.json'] = probe
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('association-calibration-probe-v1','association-calibration-fit-v1')
        source = source.replace('association_calibration_probe','association_calibration_fit')
        if index==len(nb['cells'])-1:
            source = source.replace("'--checkpoint', str(checkpoint)","'--scope', 'full', '--checkpoint', str(checkpoint)")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='association-calibration-fit-v1',scope='96 fitting/24 calibration-diagnostic original training movies',
        probe_result_sha256=report['source_sha256']['result'])
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Association Calibration Fit v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
