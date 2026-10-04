"""Stage the bounded training-only BatchNorm experiment and numerical gate."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-bn-calibration-v1'
CHECKPOINT_SHA = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-joint-selection.py'))['build'](CHECKPOINT_SHA,1)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    for name,path in [('run_recalibration.py','scripts/run-independent-bn-calibration.py'),
                      ('bn_recalibration.py','research/bn_recalibration.py'),
                      ('tests/test_bn_recalibration.py','tests/test_bn_recalibration.py')]:
        runtime[name] = (ROOT/path).read_text(encoding='utf-8')
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for cell in nb['cells']:
        source = ''.join(cell['source']).replace('independent-joint-selection-v1','independent-bn-calibration-v1')
        source = source.replace('/kaggle/working/independent_joint_selection','/kaggle/working/independent_bn_calibration')
        cell['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source']).replace("runtime/'run_selection.py'","runtime/'run_recalibration.py'")
    gate = """gate = subprocess.run([sys.executable,'-m','pytest','tests/test_bn_recalibration.py','-q'],
    cwd=runtime,capture_output=True,text=True,timeout=180)
print(gate.stdout,flush=True)
print(gate.stderr,flush=True)
if gate.returncode != 0:
    raise RuntimeError('BatchNorm numerical gate failed')
"""
    launch = launch.replace('import signal\n',gate+'import signal\n')
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-bn-calibration-v1',optimizer_steps=0,
        calibration_scope='Frozen120 training images only; no labels used for recalibration')
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent BN Calibration v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
