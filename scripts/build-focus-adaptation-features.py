"""Stage only after complete detector and label audits; replay before full cache."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_feature_spec import sparse_windows,validate_spec
SLUG='biohub-focus-adaptation-features-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_spec():
    import numpy as np
    folder=ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-cache-v1'
    receipt=runpy.run_path(str(ROOT/'scripts/verify-focus-adaptation-cache.py'))['verify'](folder)
    receipt_path=ROOT/'reports/experiments/focus-adaptation-cache-v1-result.json'
    labels_path=ROOT/'reports/experiments/focus-adaptation-labels-v1-result.json'
    if json.loads(receipt_path.read_text())!=receipt:raise ValueError('Exact completed verified raw cache required')
    labels=json.loads(labels_path.read_text())
    if (labels['status']!='completed_eight_movie_focus_adaptation_label_inventory'
        or labels['raw_cache_receipt_sha256']!=sha(receipt_path) or labels['contract']!=receipt['contract']
        or any(sha(ROOT/p)!=v for p,v in labels['source_hashes'].items())):
        raise ValueError('Complete unchanged audited training labels required')
    for role in ('fitting','diagnostic'):
        if labels['totals'][role].get('known_parent',0)<100 or labels['totals'][role].get('known_parent_absent',0)<1:
            raise ValueError('Insufficient known-parent/null coverage; do not launch features')
    probe_path=ROOT/'reports/experiments/focus-owned-neural-probe-v1-result.json'
    if sha(probe_path)!='7a0a471de64e1de38c2e40f5c8f50c75f208be1093bb08db77aa0e9d89aa71cd':
        raise ValueError('Exact existing neural probe receipt required')
    probe_root=ROOT/'.biohub/cache/kernel-outputs/focus-owned-neural-probe-v1/focus_owned_neural_probe/outputs'
    probe=json.loads((probe_root/'result.json').read_text())
    movies=[];inventory={r['stem']:r for r in labels['records']}
    for record in receipt['records']:
        stem=record['stem'];item=dict(stem=stem,role=record['role'],raw_sha256=record['sha256'])
        if record['role']=='replay':
            prior=next(r for r in probe['records'] if r['stem']==stem)
            if sha(probe_root/(stem+'.npz'))!=prior['sha256']:raise ValueError('Earlier actual GPU replay changed')
            item.update(probe_sha256=prior['sha256'],normalized_image_sha256=prior['normalized_image_sha256'])
        else:
            path=ROOT/'.biohub/cache/focus-adaptation-labels-v1'/(stem+'.json');entry=inventory[stem]
            if sha(path)!=entry['labels_sha256'] or entry['raw_sha256']!=record['sha256'] or entry['role']!=record['role']:
                raise ValueError('Training labels differ from audited raw movie/role')
            with np.load(folder/'raw_detections'/(stem+'.npz'),allow_pickle=False) as data:
                windows=sparse_windows(json.loads(path.read_text()),data['coords'],record['role'])
            item.update(labels_sha256=sha(path),windows=windows)
        movies.append(item)
    spec=dict(contract=receipt['contract'],movies=movies,raw_receipt_sha256=sha(receipt_path),
              label_receipt_sha256=sha(labels_path),probe_receipt_sha256=sha(probe_path),
              probe_model_tensor_sha256='41e82ccfd2e049abbc3d7d11e5eb08b60361b3673b606f848787597de6d62327')
    validate_spec(spec,receipt['contract']);return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-association-calibration-probe.py'))['build']()
    paths={'run_pilot.py':'scripts/collect-focus-adaptation-features.py','split.json':'research/independent_real_baseline_v1_split.json'}
    for name in ('focus_cached_pair','focus_feature_spec','focus_adaptation_cache_contract','independent_real_baseline',
                 'image_motion_residual','backward_flow_model','backward_flow_ops','independent_motion_prior','motion_residual','raw_centroid_flow_sampling'):
        paths[name+'.py']='research/'+name+'.py'
    runtime={name:(ROOT/path).read_text(encoding='utf-8') for name,path in paths.items()}
    runtime['features_spec.json']=json.dumps(spec,allow_nan=False)
    runtime['source_hashes.json']=json.dumps({name:hashlib.sha256(value.encode()).hexdigest() for name,value in runtime.items()})
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('association-calibration-probe-v1','focus-adaptation-features-v1').replace('association_calibration_probe','focus_adaptation_features')
        source=source.replace("subprocess.run([sys.executable,'-m','pytest',str(runtime/'tests'),'-q'],cwd=runtime,timeout=180,check=True)\n",'')
        if index==len(nb['cells'])-1:
            locator="""def kernel_root(slug,relative):
    candidates=[Path('/kaggle/input')/prefix/slug/relative for prefix in ('','notebooks/indarkarhana','kernels/indarkarhana')]
    found=[p for p in candidates if p.is_dir()]
    if len(found)!=1:raise RuntimeError('Unambiguous frozen kernel mount required: '+slug)
    return found[0]
raw_root=kernel_root('biohub-focus-adaptation-cache-v1','')
probe_root=kernel_root('biohub-focus-owned-neural-probe-v1','focus_owned_neural_probe')
"""
            source=source.replace('command = [',locator+'command = [')
            source=source.replace("'--checkpoint', str(checkpoint),", "'--raw-root', str(raw_root), '--probe-root', str(probe_root), '--checkpoint', str(checkpoint),")
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id='focus-adaptation-features-v1',scope='Replay first; frozen features on eight original training movies',
        contract=spec['contract'],optimizer_run=False,declared_budget_seconds=3600)
    nb['metadata']['codex'].pop('windows_per_movie',None)
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb',is_private=True,enable_internet=False,
        enable_gpu=True,enable_tpu=False,kernel_sources=['indarkarhana/biohub-image-motion-linker-v1/2',
            'indarkarhana/biohub-focus-adaptation-cache-v1/1','indarkarhana/biohub-focus-owned-neural-probe-v1/1'])
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(target)
