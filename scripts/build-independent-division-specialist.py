"""Small gated rare-division fit initialized from the improved learned linker."""
import argparse
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-division-specialist-v1'
CHECKPOINT_SHA = 'b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef'


def build(steps=100):
    if steps not in (100,1000):
        raise ValueError('Only functionality probe or bounded fit allowed')
    base = runpy.run_path(str(ROOT/'scripts/build-independent-known-null.py'))
    nb,meta = base['build'](steps)
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    runtime['run_association.py'] = runtime['run_association.py'].replace(
        "run_id='independent-known-null-v1'", "run_id='independent-division-specialist-v1'")
    for name,path in [('division_window_sampling.py','research/division_window_sampling.py'),
        ('division_missing_null_loss.py','research/division_missing_null_loss.py'),
        ('sparse_division_balanced_loss.py','research/sparse_division_balanced_loss.py'),
        ('tests/test_division_missing_null_loss.py','tests/test_division_missing_null_loss.py')]:
        runtime[name] = (ROOT/path).read_text(encoding='utf-8')
    runtime['tests/test_division_missing_null_loss.py'] = runtime['tests/test_division_missing_null_loss.py'].replace(
        "str(Path(__file__).resolve().parents[1]/'research')", "str(Path(__file__).resolve().parents[1])")
    source = source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source'] = source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source = ''.join(nb['cells'][index]['source']).replace('independent-known-null-v1','independent-division-specialist-v1')
        source = source.replace('independent_known_null','independent_division_specialist')
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    launch = ''.join(nb['cells'][-1]['source'])
    launch = launch.replace('biohub-independent-joint-broad-v1','biohub-independent-known-null-v1')
    launch = launch.replace('independent_joint_broad/outputs/last.pt','independent_known_null/outputs/last.pt')
    if launch.count(base['CHECKPOINT_SHA']) != 1:
        raise ValueError('Initialization hash marker changed')
    launch = launch.replace(base['CHECKPOINT_SHA'],CHECKPOINT_SHA)
    launch = launch.replace("'--known-null'","'--division-specialist'")
    marker = 'process = subprocess.Popen(command, start_new_session=True)'
    if launch.count(marker) != 1:
        raise ValueError('Worker launch marker changed')
    launch = launch.replace(marker,"subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests/test_division_missing_null_loss.py'),'-q'], cwd=runtime,timeout=180,check=True)\n"+marker)
    nb['cells'][-1]['source'] = launch.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='independent-division-specialist-v1',division_specialist=True,
        checkpoint_sha256=CHECKPOINT_SHA,checkpoint_version=2)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Division Specialist v1',
                code_file=TARGET.name+'.ipynb',kernel_sources=['indarkarhana/biohub-independent-known-null-v1/2'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,choices=(100,1000),default=100)
    nb,meta = build(parser.parse_args().steps)
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
