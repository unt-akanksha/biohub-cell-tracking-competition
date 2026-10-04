"""Paired source-only temporal encoder fits, followed by frozen reciprocal gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from research.native_division_context_model_v6 import TemporalDivision,paired_crops


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def guard():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(int(line)!=os.getpid() for line in text.splitlines() if line.strip()):raise RuntimeError('Foreign GPU process; leave it untouched')
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if memory['MemAvailable']<3*1024**3:raise RuntimeError('Available system RAM reserve reached')


def geometry(coords):
    delta=coords[:,1:]-coords[:,:1];r=delta.square().sum(-1).sqrt();ordered=r.sort(-1).values
    separation=(coords[:,1]-coords[:,2]).square().sum(-1).sqrt();center=delta.mean(1).square().sum(-1).sqrt()
    cosine=(delta[:,0]*delta[:,1]).sum(-1)/(r[:,0]*r[:,1]).clamp_min(1e-6)
    return torch.stack((ordered[:,0]/10,ordered[:,1]/10,separation/10,center/10,cosine,(ordered[:,1]-ordered[:,0])/10),-1).numpy()


@torch.inference_mode()
def evaluate(model,data,indices,use_context):
    model.eval();out=[]
    for first in range(0,len(indices),16):
        selected=torch.as_tensor(indices[first:first+16],device='cuda')
        original,context,coords=paired_crops(data['original'][selected],data['context'][selected],data['coords'][selected],data['valid'][selected])
        with torch.autocast('cuda',dtype=torch.float16):logits=model(original,context,coords,data['valid'][selected],use_context=use_context)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--diagnose-only',action='store_true')
    args=parser.parse_args();guard();cp=ROOT/'CONTRACT.json';contract=json.loads(cp.read_text())
    for item in contract['files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen training source changed')
    source_path=args.data/'RESULT.json'
    if sha(source_path)!='933fab6898ae726168614d28494b963a2f79c844fa400bfed608e3fe2bf0b61c':raise ValueError('Context dataset changed')
    smoke_path=Path('/dev/shm/biohub-native-division-context-v6-model-smoke/RESULT.json')
    if sha(smoke_path)!='0e595a2f954c32d33376f7343767cdac6304d74bf29eb41f42a6d1d335414053':raise ValueError('Functionality proof changed')
    smoke=json.loads(smoke_path.read_text())
    if smoke['status']!='temporal_model_smoke_passed' or smoke['median_step_seconds']*12000>1200:raise ValueError('Insufficient runtime headroom')
    args.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.75)
    started=time.monotonic();timer=threading.Timer(1800,lambda:os._exit(124));timer.daemon=True;timer.start()
    manifest=json.loads(source_path.read_text());arrays={k:[] for k in ('original','context','coords','valid','labels')};embryos=[];roles=[]
    for r in manifest['records']:
        path=args.data/r['path']
        if sha(path)!=r['sha256']:raise ValueError('Context packet changed')
        with np.load(path,allow_pickle=False) as p:
            ids=p['triples'];arrays['original'].append(p['patches'][ids]);arrays['context'].append(p['context_patches'][ids])
            arrays['coords'].append(p['coords']);arrays['valid'].append(p['context_valid'][ids]);arrays['labels'].append(p['labels'])
            embryos.extend([r['embryo']]*len(ids));roles.extend([r['role']]*len(ids))
    cpu={k:np.concatenate(v) for k,v in arrays.items()};del arrays
    labels=cpu['labels'];embryos=np.array(embryos);roles=np.array(roles)
    geom=geometry(torch.from_numpy(cpu['coords']).float())
    data={k:torch.from_numpy(v).cuda() for k,v in cpu.items()};data['coords']=data['coords'].float();data['labels']=data['labels'].float()
    del cpu
    helpers=runpy.run_path(str(ROOT/'scripts/train-division-heads-v3.py'))
    metrics=helpers['metrics'];predict=helpers['predict'];gate=helpers['gate']
    old_path=Path('/tmp/biohub-image-context-v2.ScdSdY/native-division-additive-v5-head-full/RESULT.json')
    if sha(old_path)!='89e24752a67b47e0ebca6ec89bd2fd23a65d63360e965f8c0c63fa6b575d421a':raise ValueError('Reference changed')
    previous=json.loads(old_path.read_text());encoders=json.loads((ROOT/'ENCODERS.json').read_text())
    result=dict(status='running',contract_sha256=sha(cp),data_sha256=sha(source_path),folds=[],
                successful_updates=0,target_pilot_labels_used=False,authorized_for_submission=False)
    def threshold(y,p):return float(np.float32(p[y==0].max()+1e-4))
    try:
        for source in ('44b6','6bba'):
            selection=np.flatnonzero((embryos==source)&(roles=='selection'));opposite=np.flatnonzero((embryos!=source)&(roles=='selection'))
            train=(embryos==source)&(roles=='optimization');positive=np.flatnonzero(train&(labels==1));negative=np.flatnonzero(train&(labels==0))
            reference=next(m for f in previous['folds'] if f['source']==source for m in f['members'] if m['family']=='geometry')
            reference_path=old_path.parent/reference['path']
            if sha(reference_path)!=reference['sha256']:raise ValueError('Geometry reference changed')
            geom_state=torch.load(reference_path,weights_only=True)
            geom_source=metrics(labels[selection],predict(geom_state,geom[selection]),geom_state['threshold'])
            if abs(geom_source['balanced_nll']-reference['source']['balanced_nll'])>1e-6:raise ValueError('Unchanged selection/geometry not reproduced')
            fold=dict(source=source,geometry_source=geom_source,origins=[]);chosen_probabilities=[]
            for origin_index,family in enumerate(('resnet3d','token_transformer3d')):
                member=next(e for e in encoders if e['embryo']==source and e['family']==family)
                if sha(Path(member['path']))!=member['sha256']:raise ValueError('Source encoder changed')
                arms=[];source_predictions={}
                for use_context in (False,True):
                    guard();seed=9600+(0 if source=='44b6' else 100)+origin_index
                    torch.manual_seed(seed);sampler=np.random.default_rng(seed);generator=torch.Generator(device='cuda').manual_seed(seed)
                    model=TemporalDivision(family).cuda();model.load_source_encoder(torch.load(member['path'],map_location='cuda',weights_only=True)['state_dict'])
                    anchor={name:p.detach().clone() for name,p in model.encoder.named_parameters()}
                    for p in model.encoder.parameters():p.requires_grad_(False)
                    head=list(model.temporal.parameters())+list(model.event.parameters())
                    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=2e-5),dict(params=head,lr=2e-4)],weight_decay=.01)
                    scaler=torch.amp.GradScaler('cuda');model.train();history=[];arm_started=time.monotonic()
                    arm_name=source+'-'+family+('-context' if use_context else '-control');directory=args.output/arm_name;directory.mkdir()
                    completed=0;attempts=0;skipped=0
                    if args.resume and source=='44b6' and origin_index==0 and not use_context:
                        restored=torch.load(args.resume,map_location='cuda',weights_only=True)
                        if restored['arm']!=arm_name or restored['step']!=250:raise ValueError('Only verified R1 step250 recovery is allowed')
                        model.load_state_dict(restored['state_dict']);optimizer.load_state_dict(restored['optimizer']);scaler.load_state_dict(restored['scaler'])
                        sampler.bit_generator.state=restored['sampler'];generator.set_state(restored['generator'].cpu())
                        torch.set_rng_state(restored['torch_rng'].cpu());torch.cuda.set_rng_state(restored['cuda_rng'].cpu())
                        completed=250;result['successful_updates']=250
                        for p in model.encoder.parameters():p.requires_grad_(True)
                        del restored
                    while completed<1500:
                        attempts+=1;step=completed+1
                        if attempts%25==0:
                            guard()
                            if time.monotonic()-started>1770:raise TimeoutError('Training watchdog headroom')
                        if step==201:
                            for p in model.encoder.parameters():p.requires_grad_(True)
                        index=np.concatenate((sampler.choice(positive,4),sampler.choice(negative,4)));index=torch.as_tensor(index,device='cuda')
                        original,context,coords=paired_crops(data['original'][index],data['context'][index],data['coords'][index],data['valid'][index],augment=True,generator=generator)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.float16):
                            logits=model(original,context,coords,data['valid'][index],use_context=use_context)
                            bce=F.binary_cross_entropy_with_logits(logits,data['labels'][index])
                        regularization=sum((p-anchor[name]).square().sum() for name,p in model.encoder.named_parameters())*1e-4 if step>200 else bce.new_zeros(())
                        loss=bce+regularization
                        scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
                        if not bool(torch.isfinite(norm)):
                            old_scale=scaler.get_scale()
                            diagnostic=dict(arm=arm_name,attempt=attempts,successful_step=completed,grad_norm=str(float(norm)),loss=float(loss),amp_scale=old_scale)
                            print(json.dumps(dict(event='nonfinite_gradient_skipped',**diagnostic)),flush=True)
                            if args.diagnose_only:
                                result.update(diagnostic=diagnostic,numerical_overflow_confirmed=True)
                                raise RuntimeError('Diagnostic confirmed nonfinite scaled gradient; no optimizer update applied')
                            scaler.update(new_scale=old_scale*.5);optimizer.zero_grad(set_to_none=True);skipped+=1
                            if skipped>50 or old_scale<1:raise RuntimeError('AMP cannot stabilize within bounded skips')
                            continue
                        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                        if scaler.get_scale()<old_scale:raise ValueError('Unexpected skipped finite-gradient update')
                        completed+=1
                        result['successful_updates']+=1
                        if step%250==0:
                            row=dict(step=step,bce=float(bce),anchor=float(regularization),grad_norm=float(norm),seconds=time.monotonic()-arm_started)
                            history.append(row);save(directory/'HISTORY.json',history)
                            temporary=args.output/'resume.partial'
                            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),step=step,
                                            arm=arm_name,sampler=sampler.bit_generator.state,generator=generator.get_state(),torch_rng=torch.get_rng_state(),
                                            cuda_rng=torch.cuda.get_rng_state()),temporary);temporary.replace(args.output/'resume.pt')
                            save(args.output/'PROGRESS.json',result);print(json.dumps(dict(arm=arm_name,**row)),flush=True)
                        if args.diagnose_only and completed>=500:raise RuntimeError('Diagnostic did not reproduce failure by step500')
                    probability=evaluate(model,data,selection,use_context);cutoff=threshold(labels[selection],probability)
                    checkpoint=directory/'best.pt';torch.save(dict(state_dict=model.state_dict(),origin_family=family,source=source,
                        use_context=use_context,threshold=cutoff,seed=seed,step=1500,encoder_sha256=member['sha256'],contract_sha256=sha(cp)),checkpoint)
                    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state_dict'])
                    np.testing.assert_array_equal(probability,evaluate(model,data,selection,use_context))
                    source_metrics=metrics(labels[selection],probability,cutoff)
                    arm=dict(use_context=use_context,path=checkpoint.relative_to(args.output).as_posix(),sha256=sha(checkpoint),source=source_metrics,
                             source_pass=gate(source_metrics,geom_source),opposite_opened=False,seconds=time.monotonic()-arm_started,steps=1500,
                             attempted_batches=attempts,amp_skipped_batches=skipped,recovered_updates=250 if args.resume and source=='44b6' and origin_index==0 and not use_context else 0)
                    arms.append(arm);source_predictions[use_context]=probability
                    save(directory/'RESULT.json',dict(status='arm_complete',**arm));print(json.dumps(dict(arm=arm_name,result=arm)),flush=True)
                    del model,optimizer,scaler,anchor;torch.cuda.empty_cache()
                eligible=[a for a in arms if a['source_pass']]
                origin=dict(family=family,arms=arms,selected=None)
                if eligible:
                    selected=min(eligible,key=lambda a:(a['source']['balanced_nll'],a['use_context']))
                    origin['selected']=dict(selected);save(args.output/(source+'-'+family+'-SOURCE_CHOICE.json'),origin)
                    model=TemporalDivision(family).cuda();state=torch.load(args.output/selected['path'],map_location='cuda',weights_only=True)
                    model.load_state_dict(state['state_dict']);probability=evaluate(model,data,opposite,selected['use_context'])
                    geom_opposite=metrics(labels[opposite],predict(geom_state,geom[opposite]),geom_state['threshold'])
                    score=metrics(labels[opposite],probability,selected['source']['threshold'])
                    origin['selected'].update(opposite_opened=True,opposite=score,geometry_opposite=geom_opposite,opposite_pass=gate(score,geom_opposite))
                    if origin['selected']['opposite_pass']:chosen_probabilities.append((source_predictions[selected['use_context']],probability))
                    del model;torch.cuda.empty_cache()
                fold['origins'].append(origin);print(json.dumps(dict(source=source,origin=origin)),flush=True)
            fold['ensemble']=dict(status='not_eligible_individual_gates')
            if len(chosen_probabilities)==2:
                sp=(chosen_probabilities[0][0]+chosen_probabilities[1][0])*.5;op=(chosen_probabilities[0][1]+chosen_probabilities[1][1])*.5
                cutoff=threshold(labels[selection],sp);sm=metrics(labels[selection],sp,cutoff);om=metrics(labels[opposite],op,cutoff)
                fold['ensemble']=dict(status='screen_passed' if gate(sm,geom_source) and gate(om,geom_opposite) else 'screen_rejected',
                                      source=sm,opposite=om,weights=[.5,.5],threshold=cutoff)
            result['folds'].append(fold);save(args.output/'PROGRESS.json',result)
        if result['successful_updates']!=12000:raise ValueError('Incomplete paired training schedule')
        result['status']='temporal_training_complete'
    except Exception as error:result.update(status='failed',error_type=type(error).__name__,error_message=str(error))
    finally:
        timer.cancel();result['seconds']=time.monotonic()-started;save(args.output/'RESULT.json',result)
        print(json.dumps(result),flush=True)
    return 0 if result['status']=='temporal_training_complete' else 1


if __name__=='__main__':raise SystemExit(main())
