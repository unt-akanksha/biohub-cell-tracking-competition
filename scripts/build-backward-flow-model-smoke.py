"""CPU forward/backward/reload and sparse target checks before real GPU fit."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-backward-flow-model-smoke-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-ops-smoke.py'))['build']()
    source = ''.join(nb['cells'][0]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'files')
    files = {name:(ROOT/name).read_text(encoding='utf-8') for name in (
        'research/backward_flow_ops.py','research/backward_flow_model.py',
        'tests/test_backward_flow_ops.py','tests/test_backward_flow_model.py')}
    source = source.replace(ast.get_source_segment(source,assignment),'files = '+repr(files))
    source = source.replace("str(work/'tests/test_backward_flow_ops.py')", "str(work/'tests')")
    source = source.replace('backward_flow_ops_smoke','backward_flow_model_smoke')
    ast.parse(source)
    nb['cells'][0]['source'] = source.splitlines(keepends=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Model Smoke v1',code_file=TARGET.name+'.ipynb')
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
