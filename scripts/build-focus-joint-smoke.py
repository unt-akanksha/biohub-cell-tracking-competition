"""Build an immutable four-step real-image joint encoder/head smoke."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN='focus-joint-smoke-v1';SLUG='biohub-'+RUN
BASE=runpy.run_path(str(ROOT/'scripts/build-focus-null-balanced-head.py'))
replace_once=BASE['replace_once']
PARENT_SHA='2253c9927b1ccc6ee87f75e835c44665a164ee8aec78a12bcdb1a5994b59664e'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def parent():
    path=ROOT/'kaggle/biohub-focus-null-balanced-head-v1/biohub-focus-null-balanced-head-v1.ipynb'
    if sha(path)!=PARENT_SHA:raise ValueError('Exact successful head-training environment required')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    return path,nb,source,node,ast.literal_eval(node.value)


def worker():
    original=BASE['worker']();source=original[:original.index('    def evaluate(')]
    source=replace_once(source,'    from focus_adaptation_training import SETTINGS,diagnostic_gate',
        '    from focus_adaptation_training import SETTINGS,diagnostic_gate\n    from focus_joint_probe import SETTINGS as JOINT,select_samples,ImagePairs')
    source=replace_once(source,'model.transformer.requires_grad_(True)','model.transformer.requires_grad_(True);model.unet.requires_grad_(True)')
    source=replace_once(source,"if not k.startswith('transformer.')","if k.startswith('detect_head.')")
    source=replace_once(source,'    def forward(sample,physical_only=False):',
        "    if spec['joint_settings']!=JOINT:raise ValueError('Fixed joint smoke settings required')\n    images=ImagePairs(args.data,spec['contract']['fitting_stems'])\n    def forward(sample,physical_only=False,cached=False):")
    source=replace_once(source,"        delta=(values[2][0,:,None]", "        if not physical_only and not cached:\n            encoded,_=model.encode(torch.from_numpy(images.get(sample)).unsqueeze(0).to(device))\n            grid=[v/v.new_tensor([1,4,4]) for v in values[2:4]]\n            values[:2]=[model._index_features(encoded[:,i],grid[i],masks[i]) for i in range(2)]\n        delta=(values[2][0,:,None]")
    tail='''    selected=select_samples(packets['fitting']);replays=[]
    encoder_before=tensor_hash(model.unet.state_dict());head_before=tensor_hash(model.transformer.state_dict())
    flow_before=tensor_hash(state['frozen_flow_model']);torch.cuda.reset_peak_memory_stats(0)
    with torch.no_grad():
        for sample in selected:
            p=sample['packet'];image=torch.from_numpy(images.get(sample)).unsqueeze(0).to(device)
            encoded,_=model.encode(image)
            for i,name in enumerate(('source','target')):
                native=torch.from_numpy(p[name+'_coords']).unsqueeze(0).to(device)
                grid=native/native.new_tensor([1,4,4]);mask=torch.ones(native.shape[:2],dtype=torch.bool,device=device)
                actual=model._index_features(encoded[:,i],grid,mask)[0].cpu()
                if not torch.equal(actual,torch.from_numpy(p[name+'_features'])):raise ValueError('Original image-to-indexed-feature exact replay failed')
            live,_=forward(sample);cached,_=forward(sample,cached=True)
            if not torch.equal(live,cached):raise ValueError('Original real-image/cached-logit exact replay failed')
            replays.append(dict(stem=sample['stem'],source_frame=int(p['source_frame']),pair_sha256=sample['pair_sha256'],source_nodes=len(p['source_indices']),target_nodes=len(p['target_indices']),exact_feature_and_logit_replay=True))
    print(json.dumps(dict(stage='all_four_original_image_replays_passed',pairs=replays)),flush=True)
    optimizer=torch.optim.AdamW([dict(params=model.unet.parameters(),lr=JOINT['encoder_learning_rate']),dict(params=model.transformer.parameters(),lr=JOINT['head_learning_rate'])],weight_decay=JOINT['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');steps=[]
    trainable=list(model.unet.parameters())+list(model.transformer.parameters())
    for step,sample in enumerate(selected,1):
        require_fitting_sample(sample);model.eval();model.transformer.train();optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.float16):scores,labels=forward(sample)
        loss=null_balanced_parent_loss(scores.float(),labels,null_weight)
        if not torch.isfinite(loss):raise ValueError('Nonfinite real joint loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        gradients={name:any(p.grad is not None and bool((p.grad!=0).any()) for p in module.parameters()) for name,module in [('encoder',model.unet),('head',model.transformer)]}
        if not all(gradients.values()):raise ValueError('Encoder and association must both receive nonzero gradients')
        norm=torch.nn.utils.clip_grad_norm_(trainable,JOINT['gradient_clip'],error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        if frozen_hash()!=frozen_before:raise ValueError('Frozen detector head changed')
        record=dict(step=step,stem=sample['stem'],source_frame=int(sample['packet']['source_frame']),loss=float(loss.detach()),gradient_norm=float(norm),nonzero_gradients=gradients,elapsed_seconds=time.monotonic()-started)
        steps.append(record);print(json.dumps(dict(stage='joint_smoke_step',**record)),flush=True)
    model.eval();encoder_after=tensor_hash(model.unet.state_dict());head_after=tensor_hash(model.transformer.state_dict())
    if encoder_before==encoder_after or head_before==head_after:raise ValueError('Both trained model groups must change')
    if tensor_hash(state['frozen_flow_model'])!=flow_before:raise ValueError('Frozen flow changed')
    path=args.output/'step-0004.pt'
    torch.save(dict(model=model.state_dict(),frozen_flow_model=state['frozen_flow_model'],optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),step=4,identity=spec,initial_identity=state['identity'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),path)
    with torch.no_grad():before=forward(selected[1])[0].cpu()
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if tensor_hash(restored['model'])!=tensor_hash(model.state_dict()) or tensor_hash(restored['frozen_flow_model'])!=flow_before:raise ValueError('Serialized tensors changed')
    model.load_state_dict(restored['model'],strict=True)
    with torch.no_grad():after=forward(selected[1])[0].cpu()
    if not torch.equal(before,after):raise ValueError('Exact real-image checkpoint reload failed')
    result=dict(status='completed_focus_joint_smoke',settings=JOINT,steps=steps,replays=replays,
        initial_model_sha256=initial_hash,final_model_sha256=tensor_hash(model.state_dict()),encoder_before=encoder_before,encoder_after=encoder_after,
        head_before=head_before,head_after=head_after,detector_before=frozen_before,detector_after=frozen_hash(),flow_before=flow_before,flow_after=tensor_hash(state['frozen_flow_model']),
        checkpoint=dict(file=path.name,sha256=sha(path),exact_real_image_reload=True),normalized_images=list(images.records.values()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        source_hashes=source_hashes,training_spec_sha256=sha(args.runtime/'training_spec.json'),null_weight=null_weight,
        fitting_pairs=len(packets['fitting']),diagnostic_pairs=len(packets['diagnostic']),diagnostic_evaluated=False,
        source_selection_opened=False,new_target_movies_opened=0,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
'''
    cli='\n\nif __name__'+original.split('\n\nif __name__')[1]
    result=source+tail+cli;ast.parse(result);return result


def make_spec():
    from research.focus_joint_probe import SETTINGS
    spec=BASE['make_spec']();spec.update(joint_settings=SETTINGS,design_sha256=sha(ROOT/f'reports/experiments/{RUN}-design.md'))
    return spec


def build(spec=None):
    if spec is None:spec=make_spec()
    path,nb,source,node,runtime=parent();runtime.pop('source_hashes.json')
    runtime.update({'run_pilot.py':worker(),'focus_joint_probe.py':(ROOT/'research/focus_joint_probe.py').read_text(encoding='utf-8'),'training_spec.json':json.dumps(spec,allow_nan=False)})
    runtime['source_hashes.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in runtime.items()})
    nb['cells'][1]['source']=source.replace(ast.get_source_segment(source,node),'runtime_sources = '+repr(runtime)).splitlines(keepends=True)
    for index in (0,len(nb['cells'])-1):
        source=''.join(nb['cells'][index]['source']).replace('focus-null-balanced-head-v1',RUN).replace('focus_null_balanced_head','focus_joint_smoke')
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    for cell in nb['cells']:ast.parse(''.join(cell['source']))
    nb['metadata']['codex'].update(run_id=RUN,scope='Four real-image joint encoder/head gradient and resource smoke steps; no diagnostic evaluation',parent_notebook_sha256=PARENT_SHA)
    meta=json.loads((path.parent/'kernel-metadata.json').read_text());meta.update(id='indarkarhana/'+SLUG,title=SLUG,code_file=SLUG+'.ipynb')
    return nb,meta


if __name__=='__main__':
    target=ROOT/'kaggle'/SLUG
    if target.exists():raise ValueError('Never overwrite staged or launched experiment')
    nb,meta=build();target.mkdir()
    (target/meta['code_file']).write_text(json.dumps(nb),encoding='utf-8')
    (target/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (target/'staged_identity.json').write_text(json.dumps(dict(run_id=RUN,notebook_sha256=sha(target/meta['code_file']),metadata_sha256=sha(target/'kernel-metadata.json'),builder_sha256=sha(Path(__file__)),status='staged_not_launched'),indent=2),encoding='utf-8')
    print(target)
