"""Prepare identical frozen feature extraction after new cache/labels verify."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_feature_spec import sparse_windows,validate_spec
from research.focus_extra_feature_scope import scope
RUN='focus-extra-fit-features-v1';SLUG='biohub-'+RUN
PARENT_SHA='502bcc24048eb779210f00e3d4874087dd3f94a9c2c02fa966ca9cbbef9bc8df'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_spec():
    import numpy as np
    old=runpy.run_path(str(ROOT/'scripts/build-focus-adaptation-features.py'))['make_spec']()
    raw_folder=ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-cache-v1'
    receipt=runpy.run_path(str(ROOT/'scripts/verify-focus-extra-fit-cache.py'))['verify'](raw_folder)
    receipt_path=ROOT/'reports/experiments/focus-extra-fit-cache-v1-result.json'
    if json.loads(receipt_path.read_text())!=receipt:raise ValueError('Complete verified new raw cache required')
    label_path=ROOT/'reports/experiments/focus-extra-fit-labels-v1-result.json';labels=json.loads(label_path.read_text())
    if (labels['status']!='completed_extra_fitting_label_inventory' or labels['raw_receipt_sha256']!=sha(receipt_path)
        or labels['contract']!=receipt['contract'] or any(sha(ROOT/p)!=v for p,v in labels['source_hashes'].items())):
        raise ValueError('Exact new fitting label audit required')
    counts=labels['new_fitting_totals']
    if counts.get('known_parent',0)<100 or counts.get('known_parent_absent',0)<1:
        raise ValueError('New fitting cache must actually add known-parent and known-absent supervision')
    policy=scope((ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    movies=[dict(r) for r in old['movies'] if r['role']=='replay']
    inventory={r['stem']:r for r in labels['records']}
    for record in receipt['records']:
        stem=record['stem']
        if record['role']=='replay':
            reference=next(r for r in movies if r['stem']==stem)
            if reference['raw_sha256']!=record['sha256']:raise ValueError('Actual new raw replay differs from previous feature probe')
            continue
        entry=inventory[stem];path=ROOT/'.biohub/cache/focus-extra-fit-labels-v1'/(stem+'.json')
        if entry['role']!='fitting' or entry['raw_sha256']!=record['sha256'] or sha(path)!=entry['labels_sha256']:
            raise ValueError('Fixed additional fitting label identity differs')
        with np.load(raw_folder/'raw_detections'/(stem+'.npz'),allow_pickle=False) as data:
            windows=sparse_windows(json.loads(path.read_text()),data['coords'],'fitting')
        movies.append(dict(stem=stem,role='fitting',raw_sha256=record['sha256'],labels_sha256=sha(path),windows=windows))
    spec=dict(contract=policy,movies=movies,raw_receipt_sha256=sha(receipt_path),label_receipt_sha256=sha(label_path),
        probe_receipt_sha256=old['probe_receipt_sha256'],probe_model_tensor_sha256=old['probe_model_tensor_sha256'],
        prior_feature_receipt_sha256=sha(ROOT/'reports/experiments/focus-adaptation-features-v1-result.json'))
    validate_spec(spec,policy);return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    parent=ROOT/'kaggle/biohub-focus-adaptation-features-v1/biohub-focus-adaptation-features-v1.ipynb'
    if sha(parent)!=PARENT_SHA:raise ValueError('Exact successful full feature-extraction notebook required')
    nb=json.loads(parent.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);runtime.pop('source_hashes.json')
    runtime['features_spec.json']=json.dumps(spec,allow_nan=False)
    # Keep the actual successful collector and feature math byte-identical.
    # Only its imported scope adapter changes; original modules remain intact.
    runtime['focus_adaptation_cache_contract.py']='from research.focus_extra_feature_scope import scope\n'
    runtime['research/__init__.py']=''
    for name in ('focus_adaptation_cache_contract','focus_extra_fit_scope','focus_extra_feature_scope'):
        runtime['research/'+name+'.py']=(ROOT/'research'/(name+'.py')).read_text(encoding='utf-8')
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-adaptation-features-v1',RUN).replace('focus_adaptation_features','focus_extra_fit_features')
        source=source.replace("kernel_root('biohub-focus-adaptation-cache-v1','')","kernel_root('biohub-focus-extra-fit-cache-v1','')")
        if index==len(nb['cells'])-1:
            source+='''
# Bundle only completed required artifacts to avoid hundreds of serial downloads.
import zipfile
bundle=Path('/kaggle/working/focus_extra_fit_features_bundle.zip')
members=[]
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_STORED) as archive:
    for path in sorted(work.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(work)
        required=(relative.as_posix()=='launcher_terminal.json' or relative.parts[0]=='outputs'
            or (relative.parts[0]=='runtime' and '__pycache__' not in relative.parts and path.suffix in ('.py','.json')))
        if required:
            name=path.relative_to(work.parent).as_posix();archive.write(path,name);members.append(name)
(bundle.with_suffix('.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),files=len(members),members=members),indent=2))
'''
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,scope='Identical frozen feature extraction on eight additional fitting movies',
        contract=spec['contract'],parent_notebook_sha256=PARENT_SHA,optimizer_run=False,declared_budget_seconds=3600)
    meta=json.loads((parent.parent/'kernel-metadata.json').read_text())
    meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb',kernel_sources=[
        'indarkarhana/biohub-image-motion-linker-v1/2','indarkarhana/biohub-focus-extra-fit-cache-v1/1','indarkarhana/biohub-focus-owned-neural-probe-v1/1'])
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched feature experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,
        notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),
        builder_sha256=sha(Path(__file__)),declared_budget_seconds=3600,
        status='staged_not_launched'),indent=2),encoding='utf-8');print(target)
