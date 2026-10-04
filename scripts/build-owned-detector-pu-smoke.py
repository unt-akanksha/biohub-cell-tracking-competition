"""CPU-only PU target/gradient gate; no detector training or public weights."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-owned-detector-pu-smoke-v1'


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-backward-flow-image-loss-smoke.py'))['build']()
    source=''.join(nb['cells'][0]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='files')
    files={name:(ROOT/name).read_text() for name in ('research/owned_detector_pu.py',
        'research/spotiflow_biohub/pu_targets.py','tests/test_owned_detector_pu.py','tests/test_owned_detector_pu_gradients.py')}
    source=source.replace(ast.get_source_segment(source,assignment),'files = '+repr(files))
    source=source.replace("str(work/'tests/test_backward_flow_image_loss.py')","str(work/'tests')")
    source=source.replace('backward_flow_image_loss_smoke','owned_detector_pu_smoke')
    ast.parse(source); nb['cells'][0]['source']=source.splitlines(keepends=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Owned Detector PU Smoke v1',code_file=TARGET.name+'.ipynb')
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
