"""One-hour offline paired training-frame detector TTA probe."""
import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-detector-spatial-tta-probe-v1'


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-association-calibration-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    runtime['run_pilot.py']=(ROOT/'scripts/run-detector-spatial-tta-probe.py').read_text()
    # Replace only the selected probe's test bundle, not any user filesystem files.
    runtime={k:v for k,v in runtime.items() if not k.startswith('tests/')}
    for name in ('edge_feature_tta','detector_spatial_tta'):
        runtime[name+'.py']=(ROOT/f'research/{name}.py').read_text()
    for name in ('detector_spatial_tta','image_motion_residual'):
        runtime[f'tests/test_{name}.py']=(ROOT/f'tests/test_{name}.py').read_text().replace(
            "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('association-calibration-probe-v1','detector-spatial-tta-probe-v1')
        source=source.replace('association_calibration_probe','detector_spatial_tta_probe')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='detector-spatial-tta-probe-v1',scope='Three training frames, paired native/D4 detection, fixed image flow',
        detector_views=8,linking_policy=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5))
    nb['metadata']['codex'].pop('windows_per_movie',None)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Detector Spatial TTA Probe v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
