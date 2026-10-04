"""Build full cached-detection diagnostic from the immutable successful probe."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN='focus-owned-flow-full-v1'
TARGET=ROOT/'kaggle'/('biohub-'+RUN)


def build():
    probe_folder=ROOT/'kaggle/biohub-focus-owned-flow-probe-v1'
    path=probe_folder/'biohub-focus-owned-flow-probe-v1.ipynb'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='36c761d95cb598f743d6ff6cd9172b228b2c37404ac3d5621329314bb2722abb':
        raise ValueError('Original GPU-tested probe notebook required')
    payload=(ROOT/'reports/experiments/focus-owned-flow-probe-v1-result.json').read_bytes()
    policy=runpy.run_path(str(ROOT/'research/focus_owned_flow_full_contract.py'))['receipt'](
        payload,(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    nb=json.loads(path.read_text()); meta=json.loads((probe_folder/'kernel-metadata.json').read_text())
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    runtime['probe_runtime.py']=runtime['run_pilot.py']
    runtime['run_pilot.py']=(ROOT/'scripts/run-focus-owned-flow-full.py').read_text()
    runtime['verified_probe.json']=payload.decode()
    for name in ('focus_owned_flow_full_contract','focus_linker_runtime','focus_linker_cache'):
        runtime[name+'.py']=(ROOT/f'research/{name}.py').read_text()
    runtime['verify_raw_cache.py']=(ROOT/'scripts/verify-focus-raw-detections.py').read_text().replace(
        'from research.focus_linker_runtime import','from focus_linker_runtime import')
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-owned-flow-probe-v1',RUN)
        source=source.replace('focus_owned_flow_probe','focus_owned_flow_full')
        if index==len(nb['cells'])-1:
            marker="reference = mounted('biohub-focus3d-raw-detections-v1','')"
            if source.count(marker)!=1: raise ValueError('Full input mount boundary changed')
            source=source.replace(marker,marker+"\nprobe = mounted('biohub-focus-owned-flow-probe-v1','focus_owned_flow_probe/outputs')")
            source=source.replace("'--reference',str(reference)","'--reference',str(reference),'--probe',str(probe)")
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    nb['metadata']['codex']=dict(run_id=RUN,contract=policy,declared_budget_seconds=3600,
        authorized_for_submission=False,ground_truth_opened=False)
    meta.update(id='indarkarhana/'+TARGET.name,title='Biohub Focus Owned Flow Full v1',code_file=TARGET.name+'.ipynb')
    meta['kernel_sources'].append('indarkarhana/biohub-focus-owned-flow-probe-v1/1')
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
    return nb,meta


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true')
    args=parser.parse_args(); nb,meta=build()
    if args.check:
        print(json.dumps(dict(run_id=RUN,gpu=meta['enable_gpu']))); raise SystemExit(0)
    if TARGET.exists(): raise ValueError('Refuse to overwrite full diagnostic staging')
    TARGET.mkdir(parents=True)
    (TARGET/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (TARGET/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(TARGET)
