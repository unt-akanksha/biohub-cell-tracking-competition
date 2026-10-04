"""CPU numerical gate then complete cached-movie paired motion evaluation."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-motion-prior-v1'


def build():
    base = runpy.run_path(str(ROOT/'scripts/build-independent-selection-scoring.py'))
    nb,meta = base['build']()
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    for path in ('scripts/score-independent-motion-prior.py','research/independent_motion_prior.py',
                 'tests/test_independent_motion_prior.py'):
        bundle[path] = (ROOT/path).read_text(encoding='utf-8')
    source = source.replace(ast.get_source_segment(source,assignment),'scoring_sources = '+repr(bundle))
    gate = '''gate = subprocess.run([sys.executable,'-m','pytest','tests/test_independent_motion_prior.py','-q'],
    cwd=scoring,timeout=180,text=True,capture_output=True)
print(gate.stdout,flush=True)
print(gate.stderr,flush=True)
if gate.returncode != 0:
    raise RuntimeError('Motion numerical gate failed')
'''
    source = source.replace('result = runpy.run_path',gate+'result = runpy.run_path')
    source = source.replace("scoring/'scripts/score-independent-selection.py'","scoring/'scripts/score-independent-motion-prior.py'")
    source = source.replace("work/'selection_score.json'","work/'motion_score.json'")
    nb['cells'][-1]['source'] = source.splitlines(keepends=True)
    nb['cells'][0]['source'] = ''.join(nb['cells'][0]['source']).replace(
        'independent_selection_score','independent_motion_score').splitlines(keepends=True)
    nb['metadata']['codex']['run_id'] = 'independent-motion-prior-v1'
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Motion Prior v1',
        code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
