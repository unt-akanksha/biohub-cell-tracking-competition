"""CPU transform/order gate before the real-image detector-only experiment."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-detector-spatial-tta-smoke-v1'


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-image-motion-residual-smoke.py'))['build']()
    source=''.join(nb['cells'][0]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='files')
    files=ast.literal_eval(assignment.value)
    for path in ('research/edge_feature_tta.py','research/detector_spatial_tta.py','tests/test_detector_spatial_tta.py'):
        files[path]=(ROOT/path).read_text()
    source=source.replace(ast.get_source_segment(source,assignment),'files = '+repr(files))
    source=source.replace("work/'tests/test_image_motion_residual.py'","work/'tests/test_detector_spatial_tta.py'")
    source=source.replace('image_motion_residual_smoke','detector_spatial_tta_smoke')
    ast.parse(source); nb['cells'][0]['source']=source.splitlines(keepends=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Detector Spatial TTA Smoke v1',code_file=TARGET.name+'.ipynb')
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
