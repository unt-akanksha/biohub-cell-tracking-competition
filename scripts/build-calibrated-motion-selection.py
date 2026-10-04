"""Hash-bind complete-movie evaluation to fitted training-only coefficients."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-calibrated-motion-selection-v1'


def build():
    report = json.loads((ROOT/'reports/experiments/association-calibration-fit-v1.json').read_text())
    if report['status']!='verified_calibration_not_tracking_candidate' or report['profile']!='full':
        raise ValueError('Verified full calibration fit required')
    path = ROOT/'.biohub/cache/kernel-outputs/association-calibration-fit-v1/association_calibration_fit/outputs/result.json'
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=report['source_sha256']['result']:
        raise ValueError('Calibration artifact changed')
    infer = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))
    receipt = infer['calibration_receipt'](json.loads(payload),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()),
        report['checkpoint_sha256'],hashlib.sha256(payload).hexdigest())
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-image-motion-linker-selection.py'))['build']()
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['association_calibration_result.json'] = payload.decode('utf-8')
    runtime['run_selection.py'] = runtime['run_selection.py'].replace('image-motion-linker-selection-v1','calibrated-motion-selection-v1')
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('image-motion-linker-selection-v1','calibrated-motion-selection-v1')
        source = source.replace('image_motion_linker_selection','calibrated_motion_selection')
        if index==len(nb['cells'])-1:
            source = source.replace('import signal',"command.extend(['--calibration-json',str(runtime/'association_calibration_result.json')])\nimport signal")
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='calibrated-motion-selection-v1',association_calibration=receipt)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Calibrated Motion Selection v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
