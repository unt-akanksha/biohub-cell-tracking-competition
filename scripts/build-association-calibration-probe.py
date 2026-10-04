"""Offline, hash-bound small real-image functionality probe; no submission."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-association-calibration-probe-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))['build'](100)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['run_pilot.py'] = (ROOT/'scripts/run-association-calibration.py').read_text(encoding='utf-8')
    for name in ('association_calibration','image_motion_residual','calibrated_motion_scores','backward_flow_model','backward_flow_ops',
                 'motion_residual','independent_motion_prior','annotated_missing_parent','seeded_frame_dataset'):
        runtime[name+'.py'] = (ROOT/f'research/{name}.py').read_text(encoding='utf-8')
    runtime['tests/test_association_calibration.py'] = (ROOT/'tests/test_association_calibration.py').read_text().replace(
        "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-real-pilot-v1','association-calibration-probe-v1')
        source = source.replace('independent_real_pilot','association_calibration_probe')
        if index==len(nb['cells'])-1:
            source = source.replace("'--steps', '100', ","'--checkpoint', str(checkpoint), ")
            locator = '''checkpoint = next((Path('/kaggle/input')/p/'image_motion_linker/outputs/last.pt' for p in (
    'biohub-image-motion-linker-v1','notebooks/indarkarhana/biohub-image-motion-linker-v1',
    'kernels/indarkarhana/biohub-image-motion-linker-v1')
    if (Path('/kaggle/input')/p/'image_motion_linker/outputs/last.pt').is_file()),None)
if checkpoint is None:
    raise RuntimeError('Verified frozen integrated model input missing')
subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests'),'-q'],cwd=runtime,timeout=180,check=True)
'''
            source = source.replace('command = [',locator+'command = [')
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='association-calibration-probe-v1',scope='4 fitting and2 calibration-diagnostic training movies',
        windows_per_movie=3,checkpoint_sha256=runpy.run_path(str(ROOT/'scripts/run-association-calibration.py'))['CHECKPOINT_SHA'])
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Association Calibration Probe v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
