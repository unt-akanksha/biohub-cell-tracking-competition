"""Numerical CPU gate followed by a fixed causal-linking full-movie test."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-causal-motion-selection-v1'


def build():
    nb, meta = runpy.run_path(str(ROOT/'scripts/build-independent-joint-scoring.py'))['build'](1)
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    for path in ('scripts/score-causal-motion-selection.py', 'scripts/score-independent-motion-prior.py',
                 'research/causal_motion_prior.py', 'research/independent_motion_prior.py',
                 'tests/test_causal_motion_prior.py', 'research/independent_real_baseline_v1_split.json',
                 'reports/experiments/independent-motion-persistence-training.json'):
        bundle[path] = (ROOT/path).read_text(encoding='utf-8')
    tail = source[source.index('\nimport runpy'):]
    gate = '''gate = subprocess.run([sys.executable,'-m','pytest','tests/test_causal_motion_prior.py','-q'],
    cwd=scoring,timeout=180,text=True,capture_output=True)
print(gate.stdout,flush=True)
print(gate.stderr,flush=True)
if gate.returncode != 0:
    raise RuntimeError('Causal motion numerical gate failed')
'''
    tail = tail.replace('result = runpy.run_path', gate+'result = runpy.run_path')
    tail = tail.replace("scoring/'scripts/score-independent-selection.py'", "scoring/'scripts/score-causal-motion-selection.py'")
    tail = tail.replace("work/'selection_score.json'", "work/'causal_motion_score.json'")
    nb['cells'][-1]['source'] = ('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
    bootstrap = ''.join(nb['cells'][0]['source']).replace('independent_joint_score', 'causal_motion_score')
    bootstrap = bootstrap.replace('independent-joint-scoring-v1', 'causal-motion-selection-v1')
    nb['cells'][0]['source'] = bootstrap.splitlines(keepends=True)
    nb['metadata']['codex']['run_id'] = 'causal-motion-selection-v1'
    meta.update(id='indarkarhana/'+TARGET.name, title='Biohub Causal Motion Selection v1',
                code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb, meta


if __name__ == '__main__':
    nb, meta = build()
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb), encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(TARGET)
