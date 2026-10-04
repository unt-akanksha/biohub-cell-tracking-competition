"""Fine-tune only our association transformer on verified frozen FOCUS features."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import numpy as np
    import torch
    from train_unet_transformer import UNetNodeTransformer,TemporalUNet3D
    from independent_real_baseline import install_empty_attention_guard
    from focus_cached_pair import validate_pair
    from focus_indexed_parent_loss import indexed_parent_loss,require_fitting_sample
    from focus_adaptation_training import SETTINGS,diagnostic_gate
    started=time.monotonic();torch.set_num_threads(2)
    source_hashes=json.loads((args.runtime/'source_hashes.json').read_text())
    if any(sha(args.runtime/k)!=v for k,v in source_hashes.items()):raise ValueError('Frozen training runtime changed')
    spec=json.loads((args.runtime/'training_spec.json').read_text())
    if spec['settings']!=SETTINGS or sha(args.checkpoint)!=spec['checkpoint_sha256']:
        raise ValueError('Exact initial checkpoint and declared fixed settings required')
    if torch.cuda.device_count()!=2:raise ValueError('Declared two-T4 environment required')
    torch.manual_seed(SETTINGS['seed']);random.seed(SETTINGS['seed'])
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=UNetNodeTransformer(TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda(0)
    model.load_state_dict(state['model'],strict=True);install_empty_attention_guard(model)
    model.requires_grad_(False).eval();model.transformer.requires_grad_(True)
    def tensor_hash(values):
        digest=hashlib.sha256()
        for name,value in values.items():
            digest.update(name.encode()+b'\0');digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_hash=tensor_hash(model.state_dict())
    if initial_hash!=spec['model_tensor_sha256']:raise ValueError('Exact pretrained neural tensors required')
    def frozen_hash():return tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('transformer.')})
    frozen_before=frozen_hash()
    output=args.features_root/'outputs'
    if sha(output/'result.json')!=spec['feature_worker_result_sha256']:raise ValueError('Verified feature result changed')
    feature_result=json.loads((output/'result.json').read_text())
    if sha(output/'features_spec.json')!=spec['feature_spec_file_sha256']:raise ValueError('Feature specification changed')
    feature_spec=json.loads((output/'features_spec.json').read_text())
    if feature_spec['contract']!=spec['contract']:raise ValueError('Fixed four/four movie contract required')
    packets={'fitting':[],'diagnostic':[]}
    for record in spec['feature_records']:
        if record['role']=='replay':continue
        stem=record['stem'];role=record['role'];raw=output/'raw_detections'/(stem+'.npz')
        if sha(raw)!=record['raw_sha256'] or sha(output/stem/'manifest.json')!=record['manifest_sha256']:
            raise ValueError('Verified movie artifact changed')
        with np.load(raw,allow_pickle=False) as data:coords=data['coords'].copy()
        manifest=json.loads((output/stem/'manifest.json').read_text())
        item=next(r for r in feature_spec['movies'] if r['stem']==stem)
        if role!=item['role'] or [r['file'] for r in manifest['pairs']]!=[f'{t:03d}.npz' for t in range(99)]:
            raise ValueError('Complete fixed-role pair inventory required')
        for t,pair in enumerate(manifest['pairs']):
            path=output/stem/pair['file']
            if sha(path)!=pair['sha256']:raise ValueError('Feature packet changed')
            with np.load(path,allow_pickle=False) as data:packet={k:data[k].copy() for k in data.files}
            counts=validate_pair(packet,coords);row=item['windows'][t]
            expected=np.full(len(packet['labels']),-1,np.int64)
            expected[np.asarray(row['columns'],np.int64)]=np.asarray(row['parent_rows'],np.int64)
            if not np.array_equal(packet['labels'],expected):raise ValueError('Unknown/parent/null supervision changed')
            if counts['known_parent']+counts['known_absent'] and (role=='diagnostic' or counts['source_nodes']>0):
                packets[role].append(dict(packet=packet,stem=stem,role=role,pair_sha256=pair['sha256']))
    if not all(packets.values()):raise ValueError('Nonempty disjoint fitting and diagnostic examples required')
    args.output.mkdir(parents=True,exist_ok=False)
    parameters=spec['motion_parameters'];device='cuda:0'
    scale=torch.tensor([1.625,.40625,.40625],device=device)
    mean=torch.tensor(parameters['mean_um'],device=device);variance=torch.tensor(parameters['variance_um2'],device=device)
    def forward(sample,physical_only=False):
        p=sample['packet'];ns=len(p['source_indices']);nt=len(p['target_indices'])
        values=[torch.from_numpy(p[k]).unsqueeze(0).to(device) for k in ('source_features','target_features','source_coords','target_coords','source_pos','target_pos')]
        masks=[torch.ones((1,n),dtype=torch.bool,device=device) for n in (ns,nt)]
        delta=(values[2][0,:,None]-values[3][0,None,:])*scale-torch.from_numpy(p['backward_um']).to(device)[None]-mean
        prior=-.5*(delta.square()/variance).sum(-1)
        neural=model.predict_edges(*values,*masks)[0] if ns and nt and not physical_only else torch.zeros_like(prior)
        return neural+prior,torch.from_numpy(p['labels']).to(device)
    def evaluate(physical_only=False):
        model.eval();total=dict(loss_sum=0.,known_parent=0,known_absent=0,correct_parent=0,correct_absent=0)
        with torch.no_grad():
            for sample in packets['diagnostic']:
                scores,labels=forward(sample,physical_only);ns=scores.shape[0]
                parent=(labels>=0)&(labels<ns);absent=labels==ns;n=int((parent|absent).sum())
                loss=indexed_parent_loss(scores,labels)
                prediction=torch.cat([scores,torch.full((1,len(labels)),-4.5,device=device)],dim=0).argmax(dim=0)
                total['loss_sum']+=float(loss)*n;total['known_parent']+=int(parent.sum());total['known_absent']+=int(absent.sum())
                total['correct_parent']+=int(((prediction==labels)&parent).sum());total['correct_absent']+=int(((prediction==labels)&absent).sum())
        total['nll']=total['loss_sum']/(total['known_parent']+total['known_absent']);return total
    physical=evaluate(True);initial=evaluate();print(json.dumps(dict(stage='initial_diagnostic',physical=physical,initial=initial)),flush=True)
    optimizer=torch.optim.AdamW(model.transformer.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay'])
    queue=[];losses=[];smoke=None
    def save(step):
        path=args.output/f'step-{step:04d}.pt'
        checkpoint=dict(model=model.state_dict(),frozen_flow_model=state['frozen_flow_model'],optimizer=optimizer.state_dict(),
            step=step,identity=spec,initial_identity=state['identity'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),remaining_sampling_queue=queue.copy())
        torch.save(checkpoint,path);restored=torch.load(path,map_location='cpu',weights_only=True)
        if tensor_hash(restored['model'])!=tensor_hash(model.state_dict()):raise ValueError('Saved model tensors changed')
        # Compare the same real packet after strict-loading serialized tensors.
        model.eval();before=forward(packets['fitting'][0])[0].detach().cpu()
        model.load_state_dict(restored['model'],strict=True)
        after=forward(packets['fitting'][0])[0].detach().cpu()
        if not torch.equal(before,after):raise ValueError('Real cached-pair strict checkpoint replay failed')
        return dict(file=path.name,sha256=sha(path),exact_real_logit_reload=True)
    for step in range(1,SETTINGS['steps']+1):
        if not queue:queue=list(range(len(packets['fitting'])));random.shuffle(queue)
        sample=require_fitting_sample(packets['fitting'][queue.pop()])
        model.eval();model.transformer.train();optimizer.zero_grad(set_to_none=True)
        scores,labels=forward(sample);loss=indexed_parent_loss(scores,labels)
        if not torch.isfinite(loss):raise ValueError('Nonfinite real training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.transformer.parameters(),SETTINGS['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
        if step==SETTINGS['smoke_steps']:
            if not any(p.grad is not None and bool((p.grad!=0).any()) for p in model.transformer.parameters()):raise ValueError('Small real training probe has no gradient')
            if frozen_hash()!=frozen_before:raise ValueError('Frozen encoder changed during smoke')
            smoke=save(step);print(json.dumps(dict(stage='four_step_smoke_passed',checkpoint=smoke,elapsed_seconds=time.monotonic()-started)),flush=True)
        if step%100==0:print(json.dumps(dict(step=step,mean_recent_loss=sum(losses[-100:])/len(losses[-100:]),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-started)),flush=True)
    if frozen_hash()!=frozen_before:raise ValueError('Frozen encoder/detector tensors changed')
    final=evaluate();checkpoint=save(SETTINGS['steps']);gate=diagnostic_gate(initial,physical,final)
    result=dict(status='completed_focus_head_adaptation',settings=SETTINGS,physical_diagnostic=physical,
        initial_diagnostic=initial,final_diagnostic=final,diagnostic_gate=gate,smoke=smoke,checkpoint=checkpoint,
        frozen_before=frozen_before,frozen_after=frozen_hash(),initial_model_sha256=initial_hash,final_model_sha256=tensor_hash(model.state_dict()),
        fitting_pairs=len(packets['fitting']),diagnostic_pairs=len(packets['diagnostic']),source_hashes=source_hashes,
        training_spec_sha256=sha(args.runtime/'training_spec.json'),all_diagnostic_samples_excluded_from_optimizer=True,
        raw_detections_changed=False,source_selection_opened=False,new_target_movies_opened=0,authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','manifest','data','output','checkpoint','features-root'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
