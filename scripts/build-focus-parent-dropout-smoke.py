"""New immutable four-step smoke for audited candidate-dropout supervision."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OLD=runpy.run_path(str(ROOT/'scripts/build-focus-null-balanced-head.py'))
replace_once=OLD['replace_once'];RUN='focus-parent-dropout-smoke-v1';SLUG='biohub-'+RUN
PARENT_SHA='2253c9927b1ccc6ee87f75e835c44665a164ee8aec78a12bcdb1a5994b59664e'
AUDIT_SHA='dd75e5643a84645e0bef2b6d7a3f80d09347b646fdcb61794d887437ca115225'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker():
    source=OLD['worker']()
    source=replace_once(source,'    from focus_adaptation_training import SETTINGS,diagnostic_gate',
        '    from focus_adaptation_training import SETTINGS,diagnostic_gate\n    from focus_parent_dropout import augment\n    SETTINGS=dict(SETTINGS,steps=4)')
    source=replace_once(source,"    null_weight=fitting_null_weight(**counts['fitting'])",
        "    null_weight=fitting_null_weight(**spec['combined_supervision_counts'])")
    insert='''    audit=json.loads((args.runtime/'dropout_audit.json').read_text())
    if sha(args.runtime/'dropout_audit.json')!=spec['dropout_audit_sha256']:raise ValueError('Frozen complete augmentation audit required')
    expected={(r['stem'],r['source_frame']):r for r in audit['records']}
    augmented=[]
    for sample in packets['fitting']:
        altered=augment(sample)
        if altered is None:continue
        key=(sample['stem'],int(sample['packet']['source_frame']));row=expected.pop(key)
        if altered['provenance']!=row['provenance'] or sample['pair_sha256']!=row['packet_sha256']:raise ValueError('Actual augmentation differs from host audit')
        p=altered['packet']
        if hashlib.sha256(p['labels'].tobytes()).hexdigest()!=row['labels_sha256'] or hashlib.sha256(p['source_indices'].tobytes()).hexdigest()!=row['source_ids_sha256']:raise ValueError('Augmented labels or candidate identities changed')
        if altered['provenance']['eligible_nonempty_source']:
            altered['pair_sha256']=sample['pair_sha256'];augmented.append((sample,altered))
    if expected or len(augmented)!=audit['eligible_pairs']:raise ValueError('Complete audited fitting augmentations required')
    biggest=max(augmented,key=lambda pair:len(pair[0]['packet']['source_indices'])*len(pair[0]['packet']['target_indices']))
    most_null=max(augmented,key=lambda pair:len(pair[1]['provenance']['synthetic_null_columns']))
    smoke_samples=[biggest[0],biggest[1],most_null[0],most_null[1]]
    updates=[]
'''
    source=replace_once(source,'    args.output.mkdir',insert+'    args.output.mkdir')
    source=replace_once(source,"        if not queue:queue=list(range(len(packets['fitting'])));random.shuffle(queue)\n        sample=require_fitting_sample(packets['fitting'][queue.pop()])",
        "        sample=require_fitting_sample(smoke_samples[step-1])")
    source=replace_once(source,'        optimizer.step();losses.append(float(loss.detach()))',
        "        optimizer.step();losses.append(float(loss.detach()))\n        if not any(p.grad is not None and bool((p.grad!=0).any()) for p in model.transformer.parameters()):raise ValueError('Every smoke update requires a head gradient')\n        updates.append(dict(step=step,stem=sample['stem'],source_frame=int(sample['packet']['source_frame']),pair_sha256=sample['pair_sha256'],augmentation=sample.get('provenance'),loss=float(loss.detach()),gradient_norm=float(norm)))")
    source=replace_once(source,"    final=evaluate();checkpoint=save(SETTINGS['steps']);gate=diagnostic_gate(initial,physical,final)",
        "    final=None;checkpoint=smoke;gate=dict(passed=False,scope='Four-step functionality only; no quality evaluation')")
    source=replace_once(source,"status='completed_focus_null_balanced_head',null_weight=null_weight",
        "status='completed_focus_parent_dropout_smoke',updates=updates,verified_augmentation_pairs=len(augmented),peak_allocated_bytes=torch.cuda.max_memory_allocated(0),null_weight=null_weight")
    ast.parse(source);return source


def make_spec():
    path=ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json'
    if sha(path)!=AUDIT_SHA:raise ValueError('Completed fitting audit required')
    audit=json.loads(path.read_text())
    if any(sha(ROOT/p)!=v for p,v in audit['source_hashes'].items()) or not audit['eligible_for_small_training_smoke']:raise ValueError('Actual audited augmentation changed')
    spec=OLD['make_spec']()
    if spec['contract']['fitting_stems']!=audit['training_stems']:raise ValueError('Fitting partition changed')
    counts=dict(known_parent=spec['supervision_counts']['fitting']['known_parent']+audit['eligible_retained_parent_labels'],
                known_absent=spec['supervision_counts']['fitting']['known_absent']+audit['eligible_natural_nulls']+audit['eligible_synthetic_nulls'])
    if counts!={'known_parent':20293,'known_absent':1458}:raise ValueError('Prespecified combined label counts required')
    from research.focus_null_balanced_loss import fitting_null_weight
    spec.update(settings=dict(spec['settings'],steps=4),combined_supervision_counts=counts,
        null_weight=fitting_null_weight(**counts),dropout_audit_sha256=AUDIT_SHA,
        design_sha256=sha(ROOT/f'reports/experiments/{RUN}-design.md'))
    return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    path=ROOT/'kaggle/biohub-focus-null-balanced-head-v1/biohub-focus-null-balanced-head-v1.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Frozen parent environment changed')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);runtime.pop('source_hashes.json')
    runtime.update({'run_pilot.py':worker(),'focus_parent_dropout.py':(ROOT/'research/focus_parent_dropout.py').read_text(encoding='utf-8'),
        'dropout_audit.json':(ROOT/'reports/experiments/focus-parent-dropout-v1-audit.json').read_text(encoding='utf-8'),
        'training_spec.json':json.dumps(spec,allow_nan=False)})
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-null-balanced-head-v1',RUN).replace('focus_null_balanced_head','focus_parent_dropout_smoke')
        source=source.replace('declared_budget_seconds=3600','declared_budget_seconds=900').replace('threading.Timer(3540,','threading.Timer(840,').replace('3480-','780-')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,declared_budget_seconds=900,parent_notebook_sha256=PARENT_SHA,scope='Four fitting stress updates; two original and two audited candidate-dropout; no quality claim')
    meta=json.loads((path.parent/'kernel-metadata.json').read_text());meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8');(target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
