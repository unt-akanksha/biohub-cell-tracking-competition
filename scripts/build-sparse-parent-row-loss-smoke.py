"""CPU-only numerical gate for sparse annotated-parent wrong-child supervision."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-sparse-parent-row-loss-smoke-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-sparse-parent-loss-smoke.py'))['build']()
    source = ''.join(nb['cells'][0]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'files')
    files = ast.literal_eval(assignment.value)
    for name in ('research/sparse_parent_row_loss.py','research/motion_residual.py',
                 'research/independent_motion_prior.py','tests/test_sparse_parent_row_loss.py'):
        files[name] = (ROOT/name).read_text(encoding='utf-8')
    source = source.replace(ast.get_source_segment(source,assignment),'files = '+repr(files))
    marker = source.index('\nfor name,content in files.items():')
    source = source[:marker]+source[marker:].replace('tests/test_sparse_parent_loss.py','tests/test_sparse_parent_row_loss.py')
    ast.parse(source)
    nb['cells'][0]['source'] = source.splitlines(keepends=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Sparse Parent Row Loss Smoke v1',
        code_file=TARGET.name+'.ipynb')
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb))
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(TARGET)
