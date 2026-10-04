"""Stage an offline one-hour flow-only functionality test; no launch here."""
import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='backward-flow-spatial-tta-probe-v1'
TARGET=ROOT/'kaggle'/('biohub-'+RUN)


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-detector-spatial-tta-probe.py'))['build']()
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    runtime['run_pilot.py']=(ROOT/'scripts/run-backward-flow-spatial-tta-probe.py').read_text()
    runtime['backward_flow_spatial_tta.py']=(ROOT/'research/backward_flow_spatial_tta.py').read_text()
    for name in ('backward_flow_spatial_tta','backward_flow_spatial_tta_geometry'):
        runtime[f'tests/test_{name}.py']=(ROOT/f'tests/test_{name}.py').read_text().replace(
            "str(Path(__file__).resolve().parents[1]/'research')","str(Path(__file__).resolve().parents[1])")
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('detector-spatial-tta-probe-v1',RUN)
        source=source.replace('detector_spatial_tta_probe','backward_flow_spatial_tta_probe')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex']=dict(run_id=RUN,declared_budget_seconds=3600,frames=3,training_only=True,
        detector='fixed parent D4',flow_views=8,selection_opened=False,target_audit_opened=False,
        authorized_for_submission=False)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Spatial TTA Probe v1',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    if TARGET.exists(): raise ValueError('Refuse to overwrite flow TTA staging')
    nb,meta=build(); TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
