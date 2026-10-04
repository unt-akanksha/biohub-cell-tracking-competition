"""Compose an immutable offline CPU image-only functionality notebook."""
import ast
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
SLUG='biohub-focus-registration-probe-v1'


def build():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))['build'](100)
    runtime={name:(ROOT/path).read_text(encoding='utf-8') for name,path in {
        'run_pilot.py':'scripts/probe-focus-image-registration.py',
        'transition_context.py':'research/temporal_contrastive/transition_context.py',
        'split.json':'research/independent_real_baseline_v1_split.json'}.items()}
    text=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    nb['cells'][1]['source']=text.replace(ast.get_source_segment(text,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for i in (0,len(nb['cells'])-1):
        text=''.join(nb['cells'][i]['source']).replace('independent-real-pilot-v1','focus-registration-probe-v1').replace('independent_real_pilot','focus_registration_probe')
        nb['cells'][i]['source']=text.splitlines(keepends=True)
    for c in nb['cells']: ast.parse(''.join(c['source']))
    nb['metadata']['codex'].update(run_id='focus-registration-probe-v1',gpu_used=False,
        scope='First three frames of two original training movies; no labels; no predictions',max_steps=0)
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb',enable_gpu=False,
        enable_tpu=False,enable_internet=False,is_private=True,kernel_sources=[],model_sources=[],keywords=[])
    meta.pop('machine_shape',None)
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists(): raise ValueError('Never overwrite staged/launched notebook')
    nb,meta=build(); target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
