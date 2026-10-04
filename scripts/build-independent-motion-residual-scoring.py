"""CPU scoring for the exact first complete residual-selection output version."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-independent-motion-residual-scoring-v1'


def build():
    base = runpy.run_path(str(ROOT/'scripts/build-independent-selection-scoring.py'))
    nb,meta = base['build']()
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    bundle['selection_launch.ipynb'] = (ROOT/'kaggle/biohub-independent-motion-residual-selection-v1/biohub-independent-motion-residual-selection-v1.ipynb').read_text()
    tail = source[source.index('\nimport runpy'):]
    tail = tail.replace('biohub-independent-selection-inference-v1','biohub-independent-motion-residual-selection-v1')
    tail = tail.replace("p/'independent_selection'","p/'independent_motion_residual_selection'")
    nb['cells'][-1]['source'] = ('scoring_sources = '+repr(bundle)+tail).splitlines(keepends=True)
    bootstrap = ''.join(nb['cells'][0]['source']).replace('independent_selection_score','independent_motion_residual_score')
    bootstrap = bootstrap.replace('independent-selection-scoring-v1','independent-motion-residual-scoring-v1')
    nb['cells'][0]['source'] = bootstrap.splitlines(keepends=True)
    nb['metadata']['codex']['run_id'] = 'independent-motion-residual-scoring-v1'
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Independent Motion Residual Scoring v1',
        code_file=TARGET.name+'.ipynb',kernel_sources=['indarkarhana/biohub-independent-motion-residual-selection-v1/1'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
