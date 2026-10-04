"""CPU official scorer for an immutable full specialist selection artifact."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-division-specialist-scoring-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-known-null-scoring.py'))['build']()
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    bundle['selection_launch.ipynb'] = (ROOT/'kaggle/biohub-division-specialist-selection-v1/biohub-division-specialist-selection-v1.ipynb').read_text(encoding='utf-8')
    tail = source[source.index('\nimport runpy'):]
    tail = tail.replace('biohub-independent-known-null-selection-v1','biohub-division-specialist-selection-v1')
    tail = tail.replace("p/'independent_known_null_selection'","p/'division_specialist_selection'")
    nb['cells'][-1]['source'] = ('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
    bootstrap = ''.join(nb['cells'][0]['source']).replace('independent_known_null_score','division_specialist_score')
    bootstrap = bootstrap.replace('independent-known-null-scoring-v1','division-specialist-scoring-v1')
    nb['cells'][0]['source'] = bootstrap.splitlines(keepends=True)
    nb['metadata']['codex']['run_id'] = 'division-specialist-scoring-v1'
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Division Specialist Scoring v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=['indarkarhana/biohub-division-specialist-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
