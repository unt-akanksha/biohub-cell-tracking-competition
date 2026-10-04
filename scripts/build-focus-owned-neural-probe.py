"""Immutable six-frame owned-linker experiment, excluding public linker weights."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
SLUG='biohub-focus-owned-neural-probe-v1'
HASHES={'6bba_f1fde7e0':'cf970302a726434df62ed45a55b2ef47648aaacd7d56c1b33b65b27faae2b5df',
        '6bba_23af9eeb':'6531abde5b1b10c1aa76552a9b1b2a7250ad3497db7a806796a4035398e6944b'}


def build():
    import numpy as np
    inputs=[]
    for stem,expected in HASHES.items():
        path=ROOT/'.biohub/cache/kernel-outputs/focus-owned-flow-full-v1/focus_owned_flow_full/outputs'/stem/'sampled_flow.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Exact previous raw training motion cache required')
        with np.load(path,allow_pickle=False) as data:
            selected=data['coords'][:,0]<3
            inputs.append(dict(stem=stem,source_sha256=expected,coords=data['coords'][selected].tolist(),backward_um=data['backward_um'][selected].tolist()))
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-association-calibration-probe.py'))['build']()
    runtime={name:(ROOT/path).read_text(encoding='utf-8') for name,path in {
        'run_pilot.py':'scripts/probe-focus-owned-neural.py','focus_owned_neural_links.py':'research/focus_owned_neural_links.py',
        'independent_real_baseline.py':'research/independent_real_baseline.py','independent_motion_prior.py':'research/independent_motion_prior.py',
        'split.json':'research/independent_real_baseline_v1_split.json'}.items()}
    runtime['training_inputs.json']=json.dumps(inputs,allow_nan=False)
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('association-calibration-probe-v1','focus-owned-neural-probe-v1').replace('association_calibration_probe','focus_owned_neural_probe')
        source=source.replace("subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests'),'-q'],cwd=runtime,timeout=180,check=True)\n",'')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for c in nb['cells']:ast.parse(''.join(c['source']))
    nb['metadata']['codex'].update(run_id='focus-owned-neural-probe-v1',scope='First three frames of two original training movies; raw FOCUS nodes and owned neural head',
        ground_truth_opened=False,public_linker_loaded=False,declared_budget_seconds=3600)
    nb['metadata']['codex'].pop('windows_per_movie',None)
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb',enable_gpu=True,enable_tpu=False,
        enable_internet=False,is_private=True,kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2'])
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb))
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(target)
