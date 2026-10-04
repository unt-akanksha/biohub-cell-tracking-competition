"""CPU gate for prospective frozen-flow learned-linker integration."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'kaggle/biohub-image-motion-residual-smoke-v1'


def build():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-backward-flow-ops-smoke.py'))['build']()
    source = ''.join(nb['cells'][0]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'files')
    files = {name:(ROOT/name).read_text(encoding='utf-8') for name in (
        'research/backward_flow_ops.py','research/backward_flow_model.py','research/independent_motion_prior.py','research/motion_residual.py',
        'research/image_motion_residual.py','tests/test_image_motion_residual.py')}
    source = source.replace(ast.get_source_segment(source,assignment),'files = '+repr(files))
    source = source.replace("work/'tests/test_backward_flow_ops.py'","work/'tests/test_image_motion_residual.py'")
    source = source.replace('backward_flow_ops_smoke','image_motion_residual_smoke')
    ast.parse(source)
    nb['cells'][0]['source'] = source.splitlines(keepends=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Image Motion Residual Smoke v1',code_file=TARGET.name+'.ipynb')
    return nb,meta


if __name__ == '__main__':
    nb,meta = build()
    TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
