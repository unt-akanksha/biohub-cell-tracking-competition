"""Complete-movie detector D4 with fixed, uncalibrated image-flow linking."""
import ast
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'kaggle/biohub-detector-spatial-tta-selection-v1'


def build():
    report=json.loads((ROOT/'reports/experiments/detector-spatial-tta-probe-v1.json').read_text())
    if report['status']!='verified_detector_tta_probe_not_selection':
        raise ValueError('Verified real detector probe required')
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-image-motion-linker-selection.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    for name in ('detector_spatial_tta','edge_feature_tta'):
        runtime[name+'.py']=(ROOT/f'research/{name}.py').read_text()
    runtime['run_selection.py']=runtime['run_selection.py'].replace('image-motion-linker-selection-v1','detector-spatial-tta-selection-v1')
    source=source.replace(ast.get_source_segment(source,assignment),'runtime_sources = '+repr(runtime))
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('image-motion-linker-selection-v1','detector-spatial-tta-selection-v1')
        source=source.replace('image_motion_linker_selection','detector_spatial_tta_selection')
        if index==len(nb['cells'])-1:
            start,end=source.index('reference_candidates ='),source.index('process = subprocess.Popen')
            source=source[:start]+"command.extend(['--standalone-image-flow','--detector-spatial-tta'])\n"+source[end:]
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id='detector-spatial-tta-selection-v1',detector_spatial_tta=True,
        standalone_image_flow=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5,
            flow_checkpoint_sha256='3006ee0f904640b16dd988d6404a1b3ac4f933ca68ca912d69fe4598f96d4788'),
        probe_result_sha256=report['source_sha256']['result'])
    nb['metadata']['codex'].pop('node_reference_kernel',None)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Detector Spatial TTA Selection v1',code_file=TARGET.name+'.ipynb',
        kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2'])
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    nb,meta=build(); TARGET.mkdir(parents=True,exist_ok=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
