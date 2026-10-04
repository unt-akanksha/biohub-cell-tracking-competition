"""Bounded offline real-image backward-flow probe, from random initialization."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-backward-flow-probe-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))['build'](100)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['run_pilot.py'] = (ROOT/'scripts/run-backward-flow-probe.py').read_text(encoding='utf-8')
    for name in ('backward_flow_ops','backward_flow_model','seeded_frame_dataset'):
        runtime[name+'.py'] = (ROOT/f'research/{name}.py').read_text(encoding='utf-8')
    for name in ('backward_flow_ops','backward_flow_model'):
        runtime[f'tests/test_{name}.py'] = (ROOT/f'tests/test_{name}.py').read_text(encoding='utf-8').replace(
            "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-real-pilot-v1','backward-flow-probe-v1')
        source = source.replace('independent_real_pilot','backward_flow_probe')
        source = source.replace("'--steps', '100', ",'')
        if index == len(nb['cells'])-1:
            marker = 'process = subprocess.Popen(command, start_new_session=True)'
            source = source.replace(marker,"subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests'),'-q'],cwd=runtime,timeout=180,check=True)\n"+marker)
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='backward-flow-probe-v1',max_steps=100,
        scope='96 fitting and24 diagnostic movies from original training pool',authorized_for_submission=False)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Probe v1',code_file=TARGET.name+'.ipynb',kernel_sources=[])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
