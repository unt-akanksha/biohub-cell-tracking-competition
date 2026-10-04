"""Promote verified dropout functionality to fixed 800-step head experiment."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
SMOKE=runpy.run_path(str(ROOT/'scripts/build-focus-gt-parent-dropout-smoke.py'))
BASE=runpy.run_path(str(ROOT/'scripts/build-focus-gt-parent-dropout-smoke.py'))
replace_once=BASE['replace_once'];RUN='focus-gt-parent-dropout-training-v1';SLUG='biohub-'+RUN
PARENT_SHA='b952e8a43e2c9be59c48503ef1a07ae2cfe0c07be5032a46f3839ec2dbeeac2e'
decode_runtime=SMOKE['decode_runtime']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker():
    source=BASE['worker']()
    source=replace_once(source,'SETTINGS=dict(SETTINGS,steps=4)','SETTINGS=dict(SETTINGS,steps=800)')
    source=replace_once(source,'    updates=[]',
        "    updates=[];checkpoints=[]\n    schedules={'original':packets['fitting'],'augmented':[a for _,a in augmented]}\n    queues={'original':[],'augmented':[]}")
    source=replace_once(source,'    queue=[];losses=[];smoke=None','    losses=[];smoke=None')
    source=replace_once(source,'remaining_sampling_queue=queue.copy()',"remaining_sampling_queues={k:v.copy() for k,v in queues.items()}")
    source=replace_once(source,"        sample=require_fitting_sample(smoke_samples[step-1])",
        "        stream='original' if step%2 else 'augmented'\n        if not queues[stream]:queues[stream]=list(range(len(schedules[stream])));random.shuffle(queues[stream])\n        sample=require_fitting_sample(schedules[stream][queues[stream].pop()])")
    source=replace_once(source,"            smoke=save(step);print(","            smoke=save(step);checkpoints.append(dict(step=step,checkpoint=smoke));print(")
    source=replace_once(source,"        if step%100==0:print(","        if step%100==0:checkpoints.append(dict(step=step,checkpoint=save(step)));print(")
    source=replace_once(source,"    final=None;checkpoint=smoke;gate=dict(passed=False,scope='Four-step functionality only; no quality evaluation')",
        "    final=evaluate();checkpoint=checkpoints[-1]['checkpoint'];gate=diagnostic_gate(initial,physical,final)")
    source=replace_once(source,"status='completed_focus_gt_parent_dropout_smoke',updates=updates",
        "status='completed_focus_gt_parent_dropout_training',checkpoints=checkpoints,updates=updates")
    source=replace_once(source,'print(json.dumps(result),flush=True)',"print(json.dumps({k:result[k] for k in ('status','final_diagnostic','diagnostic_gate','elapsed_seconds')}),flush=True)")
    ast.parse(source);return source


def make_spec():
    verify=runpy.run_path(str(ROOT/'scripts/verify-focus-gt-parent-dropout-smoke.py'))['verify']
    actual=verify();path=ROOT/'reports/experiments/focus-gt-parent-dropout-smoke-v1-result.json'
    if not path.exists() or json.loads(path.read_text())!=actual or not actual['eligible_for_bounded_training']:raise ValueError('Actual verified successful smoke required')
    spec,_=SMOKE['make_spec']()
    spec.update(settings=dict(spec['settings'],steps=800),verified_smoke_sha256=sha(path),
        design_sha256=sha(ROOT/f'reports/experiments/{RUN}-design.md'))
    return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    path=ROOT/'kaggle/biohub-focus-gt-parent-dropout-smoke-v1/biohub-focus-gt-parent-dropout-smoke-v1.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Successful smoke stage required')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    runtime=decode_runtime(source);runtime.pop('source_hashes.json')
    runtime['run_pilot.py']=worker();runtime['training_spec.json']=json.dumps(spec,allow_nan=False)
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    runtime.pop('dropout_audit.json')  # Existing packed literal reconstructs the exact original bytes.
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace(SMOKE['RUN'],RUN).replace('focus_gt_parent_dropout_smoke','focus_gt_parent_dropout_training')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,parent_notebook_sha256=PARENT_SHA,scope='Fixed800 alternating original/augmented fitting steps; unchanged real diagnostic gate')
    meta=json.loads((path.parent/'kernel-metadata.json').read_text());meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    if len(json.dumps(nb).encode())>=950000:raise ValueError('Notebook size bound required')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite frozen stage')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8');(target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
