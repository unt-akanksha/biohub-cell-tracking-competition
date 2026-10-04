"""Four-step real smoke using independently verified exact-GT augmentation."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke.py'))
PACK=runpy.run_path(str(ROOT/'scripts/build-focus-parent-dropout-smoke-v3.py'))
replace_once=BASE['replace_once'];decode_runtime=PACK['decode_runtime']
RUN='focus-gt-parent-dropout-smoke-v1';SLUG='biohub-'+RUN
PARENT_SHA='3b650cd894658c7fc237028cd1dd068e1931627e9fe96ba36c0eec338186cabb'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker():
    source=BASE['worker']()
    source=replace_once(source,'    from focus_parent_dropout import augment','    from focus_gt_parent_dropout import augment')
    source=replace_once(source,"        altered=augment(sample)\n        if altered is None:continue\n        key=(sample['stem'],int(sample['packet']['source_frame']));row=expected.pop(key)",
        "        key=(sample['stem'],int(sample['packet']['source_frame']));row=expected.pop(key)\n        altered=augment(sample,row['parent_gt_coords'])\n        if altered is None:raise ValueError('Every audited fitting pair has a selected parent')")
    source=replace_once(source,"status='completed_focus_parent_dropout_smoke'","status='completed_focus_gt_parent_dropout_smoke'")
    ast.parse(source);return source


def make_spec():
    verified,bundle=runpy.run_path(str(ROOT/'scripts/verify-focus-gt-parent-dropout.py'))['verify']()
    path=ROOT/'reports/experiments/focus-gt-parent-dropout-v1-verification.json'
    if sha(path)!='b8125cf3f1c2a6431e0f8d3ee48934a50dc67438582aadce12422c8aca2fa7c7' or json.loads(path.read_text())!=verified:raise ValueError('Actual full exact-GT replay receipt required')
    spec=BASE['make_spec']()
    counts=dict(known_parent=spec['supervision_counts']['fitting']['known_parent']+bundle['eligible_retained_parent_labels'],
                known_absent=spec['supervision_counts']['fitting']['known_absent']+bundle['eligible_natural_nulls']+bundle['eligible_synthetic_nulls'])
    if counts!={'known_parent':20387,'known_absent':1443} or bundle['training_stems']!=spec['contract']['fitting_stems']:raise ValueError('Exact fitting scope and counts required')
    from research.focus_null_balanced_loss import fitting_null_weight
    blob=json.dumps(bundle,separators=(',',':'),allow_nan=False)
    spec.update(combined_supervision_counts=counts,null_weight=fitting_null_weight(**counts),
        dropout_audit_sha256=hashlib.sha256(blob.encode()).hexdigest(),gt_dropout_verification_sha256=sha(path),
        design_sha256=sha(ROOT/f'reports/experiments/{RUN}-design.md'))
    return spec,bundle


def build(spec=None,bundle=None):
    if spec is None:spec,bundle=make_spec()
    if bundle is None:raise ValueError('Explicit verified or test bundle required')
    path=ROOT/'kaggle/biohub-focus-parent-dropout-smoke-v3/biohub-focus-parent-dropout-smoke-v3.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Frozen successful smoke environment required')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source']);runtime=decode_runtime(source);runtime.pop('source_hashes.json')
    blob=json.dumps(bundle,separators=(',',':'),allow_nan=False)
    if spec.get('dropout_audit_sha256',hashlib.sha256(blob.encode()).hexdigest())!=hashlib.sha256(blob.encode()).hexdigest():raise ValueError('Exact emitted bundle hash required')
    runtime.update({'run_pilot.py':worker(),'focus_gt_parent_dropout.py':(ROOT/'research/focus_gt_parent_dropout.py').read_text(encoding='utf-8'),
                    'dropout_audit.json':blob,'training_spec.json':json.dumps(spec,allow_nan=False)})
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    packed=base64.b64encode(gzip.compress(runtime.pop('dropout_audit.json').encode(),mtime=0)).decode()
    replacements=[]
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id=='runtime_sources':replacements.append((ast.get_source_segment(source,n),'runtime_sources = '+repr(runtime)))
            if n.targets[0].id=='packed_dropout_audit':replacements.append((ast.get_source_segment(source,n),'packed_dropout_audit = '+repr(packed)))
    for before,after in replacements:source=source.replace(before,after)
    if decode_runtime(source)['dropout_audit.json']!=blob:raise ValueError('Lossless exact-GT metadata required')
    nb['cells'][1]['source']=source.splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace(PACK['RUN'],RUN).replace('focus_parent_dropout_smoke_v3','focus_gt_parent_dropout_smoke')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,parent_notebook_sha256=PARENT_SHA,scope='Four-step verified exact-GT candidate-dropout functionality; no quality claim')
    meta=json.loads((path.parent/'kernel-metadata.json').read_text());meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    if len(json.dumps(nb).encode())>=950000:raise ValueError('Source size bound required')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite a frozen stage')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8');(target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
