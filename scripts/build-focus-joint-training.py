"""Live-image joint training after verified stress/AMP smoke, bounded30min."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
SMOKE=runpy.run_path(str(ROOT/'scripts/build-focus-joint-backoff-smoke.py'));replace_once=SMOKE['replace_once']
RUN='focus-joint-training-v1';SLUG='biohub-'+RUN
PARENT_SHA='d94541b02fd1ef17733759975bcabf08fbf99cb308133451c3974b088e9d94c3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker():
    smoke=SMOKE['worker']();source=smoke[:smoke.index("    selected=select_samples(packets['fitting'])")]
    source=replace_once(source,'    from focus_joint_probe_backoff import SETTINGS as JOINT,joint_update',
        '    from focus_joint_probe_backoff import joint_update\n    from focus_joint_training import SETTINGS as JOINT')
    source=replace_once(source,"images=ImagePairs(args.data,spec['contract']['fitting_stems'])","images=ImagePairs(args.data,spec['contract']['fitting_stems']+spec['contract']['diagnostic_stems'])")
    old=SMOKE['OLD']['BASE']['worker']()
    evaluate=old[old.index('    def evaluate('):old.index('    physical=evaluate(True)')]
    checks=old[old.index('    physical=evaluate(True)'):old.index('    optimizer=torch.optim.AdamW')]
    preflight='''    encoder_before=tensor_hash(model.unet.state_dict());head_before=tensor_hash(model.transformer.state_dict())
    flow_before=tensor_hash(state['frozen_flow_model']);torch.cuda.reset_peak_memory_stats(0)
    selected=select_samples(packets['fitting']);preflight=[]
    with torch.no_grad():
        for sample in selected:
            live,_=forward(sample);cached,_=forward(sample,cached=True)
            if not torch.equal(live,cached):raise ValueError('Initial real-image/cached-logit replay failed')
            preflight.append(dict(stem=sample['stem'],source_frame=int(sample['packet']['source_frame']),pair_sha256=sample['pair_sha256'],exact_logit_replay=True))
'''
    tail='''    optimizer=torch.optim.AdamW([dict(params=model.unet.parameters(),lr=JOINT['encoder_learning_rate']),dict(params=model.transformer.parameters(),lr=JOINT['head_learning_rate'])],weight_decay=JOINT['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=JOINT['amp_initial_scale']);queue=[];updates=[];checkpoints=[]
    def save(step):
        path=args.output/f'step-{step:04d}.pt'
        checkpoint=dict(model=model.state_dict(),frozen_flow_model=state['frozen_flow_model'],optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),step=step,identity=spec,initial_identity=state['identity'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),remaining_sampling_queue=queue.copy())
        torch.save(checkpoint,path);restored=torch.load(path,map_location='cpu',weights_only=True)
        if tensor_hash(restored['model'])!=tensor_hash(model.state_dict()) or tensor_hash(restored['frozen_flow_model'])!=flow_before:raise ValueError('Serialized checkpoint tensors changed')
        model.eval()
        with torch.no_grad():before=forward(selected[1])[0].cpu()
        model.load_state_dict(restored['model'],strict=True)
        with torch.no_grad():after=forward(selected[1])[0].cpu()
        if not torch.equal(before,after):raise ValueError('Exact real-image checkpoint reload failed')
        row=dict(step=step,file=path.name,sha256=sha(path),exact_real_image_reload=True);checkpoints.append(row);return row
    for step in range(1,JOINT['steps']+1):
        if not queue:queue=list(range(len(packets['fitting'])));random.shuffle(queue)
        sample=require_fitting_sample(packets['fitting'][queue.pop()])
        update=joint_update(model,optimizer,scaler,sample,forward,null_balanced_parent_loss,null_weight,tensor_hash)
        if frozen_hash()!=frozen_before or tensor_hash(state['frozen_flow_model'])!=flow_before:raise ValueError('Frozen detector or flow changed')
        update.update(step=step,stem=sample['stem'],source_frame=int(sample['packet']['source_frame']));updates.append(update)
        saved=None
        if step==4 or step%JOINT['checkpoint_interval']==0:
            saved=save(step)
            print(json.dumps(dict(stage='joint_training_checkpoint',checkpoint=saved,mean_recent_loss=sum(r['loss'] for r in updates[-100:])/len(updates[-100:]),overflow_skips=sum(len(r['overflow_retries']) for r in updates),elapsed_seconds=time.monotonic()-started)),flush=True)
        if time.monotonic()-started>JOINT['worker_training_stop_seconds']:
            if saved is None:saved=save(step)
            result=dict(status='budget_checkpointed_incomplete_joint_training',completed_steps=step,checkpoint=saved,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
            (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    final=evaluate();gate=diagnostic_gate(initial,physical,final)
    encoder_after=tensor_hash(model.unet.state_dict());head_after=tensor_hash(model.transformer.state_dict())
    if encoder_before==encoder_after or head_before==head_after:raise ValueError('Both trained groups must change')
    result=dict(status='completed_focus_joint_training',settings=JOINT,initial_model_sha256=initial_hash,final_model_sha256=tensor_hash(model.state_dict()),
        encoder_before=encoder_before,encoder_after=encoder_after,head_before=head_before,head_after=head_after,detector_before=frozen_before,detector_after=frozen_hash(),flow_before=flow_before,flow_after=tensor_hash(state['frozen_flow_model']),
        physical_diagnostic=physical,initial_diagnostic=initial,final_diagnostic=final,diagnostic_gate=gate,preflight=preflight,checkpoints=checkpoints,updates=updates,normalized_images=list(images.records.values()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        source_hashes=source_hashes,training_spec_sha256=sha(args.runtime/'training_spec.json'),null_weight=null_weight,
        fitting_pairs=len(packets['fitting']),diagnostic_pairs=len(packets['diagnostic']),all_diagnostic_samples_excluded_from_optimizer=True,
        source_selection_opened=False,new_target_movies_opened=0,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(status=result['status'],final_diagnostic=final,diagnostic_gate=gate,elapsed_seconds=result['elapsed_seconds'])),flush=True)
'''
    cli='\n\nif __name__'+smoke.split('\n\nif __name__')[1]
    result=source+evaluate+preflight+checks+tail+cli;ast.parse(result);return result


def make_spec():
    from focus_joint_training import SETTINGS
    verified=runpy.run_path(str(ROOT/'scripts/verify-focus-joint-backoff-smoke.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-joint-backoff-smoke-v1')
    receipt=ROOT/'reports/experiments/focus-joint-backoff-smoke-v1-result.json'
    if json.loads(receipt.read_text())!=verified:raise ValueError('Actual verified successful joint smoke required')
    path=ROOT/'kaggle/biohub-focus-joint-backoff-smoke-v1/biohub-focus-joint-backoff-smoke-v1.ipynb';nb=json.loads(path.read_text())
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    spec=json.loads(ast.literal_eval(node.value)['training_spec.json'])
    spec.update(joint_settings=SETTINGS,design_sha256=sha(ROOT/f'reports/experiments/{RUN}-design.md'),joint_smoke_receipt_sha256=sha(receipt),declared_budget_seconds=1800)
    return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    path=ROOT/'kaggle/biohub-focus-joint-backoff-smoke-v1/biohub-focus-joint-backoff-smoke-v1.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Exact verified joint-smoke notebook required')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);runtime.pop('source_hashes.json')
    runtime.update({'run_pilot.py':worker(),'focus_joint_training.py':(ROOT/'research/focus_joint_training.py').read_text(encoding='utf-8'),'training_spec.json':json.dumps(spec,allow_nan=False)})
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-joint-backoff-smoke-v1',RUN).replace('focus_joint_backoff_smoke','focus_joint_training')
        source=source.replace('3600','1800').replace('3480','1680')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,scope='800 live-image joint encoder/head steps; unchanged diagnostic gate',parent_notebook_sha256=PARENT_SHA,declared_budget_seconds=1800)
    meta=json.loads((path.parent/'kernel-metadata.json').read_text());meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8');(target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
