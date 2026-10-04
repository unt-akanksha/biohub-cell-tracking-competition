"""Three-arm real smoke: native, zero-weight shortcut, then flow averaging.

V1 staging is preserved and must not be launched for this expanded test.
"""
import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
RUN='backward-flow-spatial-tta-probe-v2'
TARGET=ROOT/'kaggle'/('biohub-'+RUN)


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-backward-flow-spatial-tta-probe.py'))['build']()
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('backward-flow-spatial-tta-probe-v1',RUN)
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex'].update(run_id=RUN,arms=['native','zero_weight_shortcut','flow_d4'],
        zero_weight_graph_parity_required=True)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Backward Flow Spatial TTA Probe v2',code_file=TARGET.name+'.ipynb')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    if TARGET.exists(): raise ValueError('Refuse to overwrite three-arm flow probe staging')
    nb,meta=build(); TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
