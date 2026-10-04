"""Immutable numerical repair of joint smoke, retaining original experiment."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
OLD=runpy.run_path(str(ROOT/'scripts/build-focus-joint-smoke.py'));replace_once=OLD['replace_once']
RUN='focus-joint-backoff-smoke-v1';SLUG='biohub-'+RUN
PARENT_SHA='87f68814914a35dfa71b2e4015f8ddf1e392b7e68d8be42654dee2054c0f1f30'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker():
    source=OLD['worker']()
    source=replace_once(source,'    from focus_joint_probe import SETTINGS as JOINT,select_samples,ImagePairs',
        '    from focus_joint_probe import select_samples,ImagePairs\n    from focus_joint_probe_backoff import SETTINGS as JOINT,joint_update')
    source=replace_once(source,"scaler=torch.amp.GradScaler('cuda')","scaler=torch.amp.GradScaler('cuda',init_scale=JOINT['amp_initial_scale'])")
    start=source.index('        require_fitting_sample(sample);model.eval()')
    end=source.index('        steps.append(record);',start)
    source=source[:start]+'''        require_fitting_sample(sample)
        record=joint_update(model,optimizer,scaler,sample,forward,null_balanced_parent_loss,null_weight,tensor_hash)
        if frozen_hash()!=frozen_before:raise ValueError('Frozen detector head changed')
        record.update(step=step,stem=sample['stem'],source_frame=int(sample['packet']['source_frame']),elapsed_seconds=time.monotonic()-started)
'''+source[end:]
    source=replace_once(source,"status='completed_focus_joint_smoke'","status='completed_focus_joint_backoff_smoke'")
    ast.parse(source);return source


def make_spec():
    from focus_joint_probe_backoff import SETTINGS
    spec=OLD['make_spec']();spec.update(joint_settings=SETTINGS,design_sha256=sha(ROOT/f'reports/experiments/{RUN}-design.md'))
    return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    path=ROOT/'kaggle/biohub-focus-joint-smoke-v1/biohub-focus-joint-smoke-v1.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Frozen failed smoke environment required')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);runtime.pop('source_hashes.json')
    runtime.update({'run_pilot.py':worker(),'focus_joint_probe_backoff.py':(ROOT/'research/focus_joint_probe_backoff.py').read_text(encoding='utf-8'),'training_spec.json':json.dumps(spec,allow_nan=False)})
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-joint-smoke-v1',RUN).replace('focus_joint_smoke','focus_joint_backoff_smoke')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,scope='Same four real joint updates with bounded same-example AMP backoff',parent_notebook_sha256=PARENT_SHA)
    meta=json.loads((path.parent/'kernel-metadata.json').read_text());meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8');(target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
