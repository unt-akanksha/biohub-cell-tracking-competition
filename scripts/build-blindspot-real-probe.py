"""Stage one immutable real-image probe only after exact CPU functionality."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
SLUG='biohub-blindspot-real-probe-v1'
CPU_SHA='6b0d19df5aae896f00650b336980904b0a71ba88a98165616f89275099d245fc'


def build():
    cpu_path=ROOT/'.biohub/cache/kernel-outputs/blindspot-cpu-probe-v1/blindspot_cpu_probe.json'
    if hashlib.sha256(cpu_path.read_bytes()).hexdigest()!=CPU_SHA:
        raise ValueError('Exact successful CPU receipt required')
    cpu=json.loads(cpu_path.read_text())
    model=(ROOT/'research/blindspot_restoration.py').read_text()
    if hashlib.sha256(model.encode()).hexdigest()!=cpu['source_sha256']['blindspot_restoration.py']:
        raise ValueError('CPU-tested model source changed')
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-independent-real-pilot.py'))['build'](100)
    runtime={'run_pilot.py':(ROOT/'scripts/run-blindspot-real-probe.py').read_text(),
        'blindspot_restoration.py':model,'blindspot_training_contract.py':(ROOT/'research/blindspot_training_contract.py').read_text(),
        'cpu_probe.json':cpu_path.read_text(),'split.json':(ROOT/'research/independent_real_baseline_v1_split.json').read_text()}
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for i in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][i]['source']).replace('independent-real-pilot-v1','blindspot-real-probe-v1').replace('independent_real_pilot','blindspot_real_probe')
        nb['cells'][i]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id='blindspot-real-probe-v1',cpu_probe_sha256=CPU_SHA,
        scope='96 fitting and24 diagnostic training-pool movies; images only; no tracking selection',declared_budget_seconds=3600)
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb',enable_gpu=True,
                enable_tpu=False,enable_internet=False,is_private=True,kernel_sources=[],model_sources=[])
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists(): raise ValueError('Do not overwrite a staged or launched probe')
    nb,meta=build(); target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
