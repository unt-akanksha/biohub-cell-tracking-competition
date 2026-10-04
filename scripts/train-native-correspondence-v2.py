"""Sequential native visual experts with source-only selection and hard budgets."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.native_correspondence_models_v2 import VisualCorrespondence, prepare_patches, mask_scores
from research.visual_correspondence_data_v1 import validate_roles


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''): digest.update(chunk)
    return digest.hexdigest()


def save(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temp.replace(path)


def pids():
    text=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout
    return {int(s.strip()) for s in text.splitlines() if s.strip()}


def load_data(folder,records,embryo,role):
    selected=[r for r in records if r['embryo']==embryo and r['role']==role]
    expected=sum(r['patches']*3*15**3*2 for r in selected)
    if expected>5*1024**3:
        raise RuntimeError('Declared resident data cap exceeded')
    patches=[]; parts={k:[] for k in ('ids','coords','valid','loss_mask','targets')}
    names=[]; movie_ids=[]; offset=0
    for record in selected:
        path=folder/record['path']
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Data artifact mismatch')
        with np.load(path,allow_pickle=False) as packet:
            stored=packet['patches']; ids=packet['ids'].copy()+offset
            if stored.shape!=(record['patches'],3,15,15,15) or ids.shape!=(record['groups'],17):
                raise ValueError('Native packet shape mismatch')
            patches.append(torch.from_numpy(stored).to('cuda')); offset+=len(stored)
            for key in parts: parts[key].append(torch.from_numpy(ids if key=='ids' else packet[key]).to('cuda'))
        if record['stem'] not in names: names.append(record['stem'])
        movie_ids.extend([names.index(record['stem'])]*record['groups'])
    # Chunk tensors plus concatenation need at most twice the declared resident bytes.
    if torch.cuda.mem_get_info()[0] < expected+6*1024**3:
        raise RuntimeError('Insufficient GPU model/concatenation reserve')
    result={key:torch.cat(values) for key,values in parts.items()}
    result['patches']=torch.cat(patches)
    result['movie_ids']=torch.tensor(movie_ids,device='cuda'); result['movie_names']=names
    result['weights']=torch.bincount(result['movie_ids'])[result['movie_ids']].float().reciprocal()
    return result


def batch(data,rows,augment=False,missing=False,generator=None):
    ids=data['ids'][rows]; valid=data['valid'][rows].clone()
    mask=data['loss_mask'][rows].clone(); targets=data['targets'][rows].clone()
    if missing or augment:
        chosen=targets<16
        if augment: chosen &= torch.rand(len(rows),device='cuda',generator=generator)<.2
        active=chosen.nonzero().flatten()
        valid[active,targets[active]+1]=False; mask[active,targets[active]]=False; targets[active]=16
    images,coords=prepare_patches(data['patches'][ids],data['coords'][rows],valid,augment,generator)
    return images,coords,valid,mask,targets


@torch.inference_mode()
def evaluate(model,data,missing=False):
    if model is not None: model.eval()
    scores=[]; labels=[]
    for start in range(0,len(data['targets']),12):
        rows=torch.arange(start,min(start+12,len(data['targets'])),device='cuda')
        x,coord,valid,mask,target=batch(data,rows,missing=missing)
        if model is None:
            distance=((coord[:,1:]-coord[:,:1])/10).square().sum(-1)
            raw=torch.cat((-2*distance,torch.full_like(distance[:,:1],-8)),dim=-1)
            raw=raw.masked_fill(~torch.cat((valid[:,1:],valid[:,:1]),dim=-1),-1e4)
        else:
            with torch.autocast('cuda',dtype=torch.float16): raw=model(x,coord,valid)
        raw=mask_scores(raw,mask).float()
        if not torch.isfinite(raw).all(): raise ValueError('Nonfinite evaluation')
        scores.append(raw.cpu()); labels.append(target.cpu())
    score=torch.cat(scores); target=torch.cat(labels); loss=F.cross_entropy(score,target,reduction='none')
    correct=score.argmax(-1)==target; movie=data['movie_ids'].cpu()
    return dict(nll=float(loss.mean()),correct=int(correct.sum()),total=len(target),present=int((target<16).sum()),
                null=int((target==16).sum()),per_movie={name:dict(total=int((movie==i).sum()),
                correct=int(correct[movie==i].sum()),nll=float(loss[movie==i].mean())) for i,name in enumerate(data['movie_names'])})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-proof',type=Path); args=parser.parse_args()
    contract=json.loads((ROOT/'TRAINING.json').read_text()); code_sha=sha(ROOT/'TRAINING.json')
    for record in contract['files']:
        if sha(ROOT/record['path'])!=record['sha256']: raise ValueError('Training code changed')
    result_path=args.data/'RESULT.json'; data=json.loads(result_path.read_text())
    if data['status']!='data_complete' or data['movie_plan_sha256']!='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6':
        raise ValueError('Require complete frozen native dataset')
    validate_roles(data['records'])
    if pids(): raise RuntimeError('GPU occupied; do not interrupt other projects')
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/tmp/biohub-image-context-v2.ScdSdY')):
        raise ValueError('Require new owned output')
    if shutil.disk_usage(args.output.parent).free<850*1024**2: raise RuntimeError('Checkpoint disk reserve unavailable')
    if not args.smoke:
        proof=json.loads(args.smoke_proof.read_text())
        if proof['status']!='smoke_passed' or proof['training_contract_sha256']!=code_sha or proof['data_result_sha256']!=sha(result_path):
            raise ValueError('Exact model/data smoke proof missing')
        if proof['projected_full_seconds']>5100: raise RuntimeError('Full training projection exceeds budget')
    args.output.mkdir(); begin=time.monotonic(); stop=threading.Event()
    result=dict(run_id='native-correspondence-v2',status='running',smoke=args.smoke,
                training_contract_sha256=code_sha,data_result_sha256=sha(result_path),members=[],
                environment=dict(torch=torch.__version__,python=sys.version,gpu=torch.cuda.get_device_name()),
                authorized_for_submission=False)
    def monitor():
        while not stop.wait(30):
            try:
                if pids()-{os.getpid()}: result['foreign_gpu_detected']=True; stop.set()
            except Exception: result['gpu_guard_failed']=True; stop.set()
    threading.Thread(target=monitor,daemon=True).start()
    watchdog=threading.Timer(900 if args.smoke else 5400,lambda:os._exit(124)); watchdog.daemon=True; watchdog.start()
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.9)
    # Sparse groups produce changing encoder batch sizes. Autotuning each new
    # size caused 0.2-0.3s spikes vs ~0.02-0.03s ordinary updates in the smoke.
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False
    cache={}; timings=[]; train=selection=None
    pairs=[(e,f) for e in ('6bba','44b6') for f in ('resnet3d','token_transformer3d')]
    if args.smoke: pairs=pairs[:2]
    try:
        for index,(embryo,family) in enumerate(pairs):
            if stop.is_set(): raise RuntimeError('Yield before member')
            member_start=time.monotonic()
            if embryo not in cache:
                train=selection=None
                cache.clear(); gc.collect(); torch.cuda.empty_cache()
                cache[embryo]=(load_data(args.data,data['records'],embryo,'optimization'),load_data(args.data,data['records'],embryo,'selection'))
            train,selection=cache[embryo]
            seed=20260914+index; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            generator=torch.Generator(device='cuda').manual_seed(seed+971)
            model=VisualCorrespondence(family,input_channels=3).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=2048.)
            baseline=evaluate(None,selection); missing_baseline=evaluate(None,selection,True)
            initial=evaluate(model,selection)
            if abs(initial['nll']-baseline['nll'])>1e-4: raise ValueError('Zero-residual prior mismatch')
            member=args.output/(embryo+'-'+family); member.mkdir()
            summary=dict(embryo=embryo,family=family,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                         baseline=baseline,missing_baseline=missing_baseline,initial=initial,history=[])
            best=float('inf'); optimizer_steps=0; max_steps=50 if args.smoke else 8000; tick=time.monotonic()
            steady_seconds=[]
            for step in range(1,max_steps+1):
                if stop.is_set() or time.monotonic()-begin>(780 if args.smoke else 5100): break
                if args.smoke: torch.cuda.synchronize()
                update_start=time.monotonic()
                model.train(); rows=torch.multinomial(train['weights'],12,replacement=True,generator=generator)
                x,coord,valid,mask,target=batch(train,rows,augment=True,generator=generator)
                optimizer.zero_grad(set_to_none=True)
                lr=2e-6+(2e-4-2e-6)*.5*(1+math.cos(math.pi*(step-1)/max_steps))
                for group in optimizer.param_groups: group['lr']=lr
                with torch.autocast('cuda',dtype=torch.float16): loss=F.cross_entropy(mask_scores(model(x,coord,valid),mask),target)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
                old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                optimizer_steps+=int(scaler.get_scale()>=old_scale)
                if args.smoke:
                    torch.cuda.synchronize()
                    if step>20: steady_seconds.append(time.monotonic()-update_start)
                if step%500==0 or step==max_steps:
                    eval_start=time.monotonic(); score=evaluate(model,selection); missing=evaluate(model,selection,True)
                    entry=dict(step=step,optimizer_steps=optimizer_steps,selection=score,missing=missing,eval_seconds=time.monotonic()-eval_start)
                    summary['history'].append(entry)
                    if score['nll']<best:
                        best=score['nll']; summary.update(best_step=step,selected=score,selected_missing=missing)
                        temp=member/'best.partial'; torch.save(dict(state_dict=model.state_dict(),family=family,input_channels=3,seed=seed,step=step,
                                                    training_contract_sha256=code_sha,data_result_sha256=sha(result_path)),temp); temp.replace(member/'best.pt')
                    temp=args.output/'resume.partial'; torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                          torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),generator_rng=generator.get_state(),step=step,
                          embryo=embryo,family=family,training_contract_sha256=code_sha,data_result_sha256=sha(result_path)),temp); temp.replace(args.output/'resume.pt')
                    save(member/'HISTORY.json',summary)
                    print(json.dumps(dict(embryo=embryo,family=family,step=step,nll=score['nll'],correct=score['correct'],total=score['total'])),flush=True)
            elapsed=time.monotonic()-tick
            if optimizer_steps!=max_steps: raise RuntimeError('Incomplete optimizer updates or external-yield request')
            checkpoint=torch.load(member/'best.pt',map_location='cuda',weights_only=True)
            model.load_state_dict(checkpoint['state_dict']); reloaded=evaluate(model,selection)
            if abs(reloaded['nll']-summary['selected']['nll'])>1e-7 or reloaded['correct']!=summary['selected']['correct']:
                raise ValueError('Checkpoint reload changed predictions')
            score=summary['selected']; missing=summary['selected_missing']
            summary.update(optimizer_steps=optimizer_steps,elapsed_seconds=time.monotonic()-member_start,weights_sha256=sha(member/'best.pt'),
                source_gate=score['nll']<=.98*baseline['nll'] and score['correct']>=baseline['correct'] and missing['nll']<=missing_baseline['nll'],
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),status='smoke_completed' if args.smoke else 'trained')
            save(member/'RESULT.json',summary); result['members'].append(summary)
            if args.smoke:
                eval_seconds=summary['history'][-1]['eval_seconds']
                update_seconds=max(float(np.mean(steady_seconds))*1.25,float(np.quantile(steady_seconds,.95)))
                timings.append(dict(family=family,update_seconds=update_seconds,paired_eval_seconds=eval_seconds,
                                    steady_update_seconds=steady_seconds,warmup_updates=20,
                                    timing_policy='max(mean*1.25,p95), CUDA synchronized, excludes validation/checkpoint I/O'))
            save(args.output/'PROGRESS.json',result)
            del model,optimizer,scaler,checkpoint; gc.collect(); torch.cuda.empty_cache()
        result['status']='smoke_passed' if args.smoke else 'training_complete'
        if args.smoke:
            result['timings']=timings
            result['projected_full_seconds']=2*sum(t['update_seconds']*8000+t['paired_eval_seconds']*16 for t in timings)+240
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__)
    finally:
        stop.set(); watchdog.cancel(); result['elapsed_seconds']=time.monotonic()-begin
        save(args.output/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='members'}),flush=True)
    return 0 if result['status'] in ('smoke_passed','training_complete') else 1


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(status='preflight_failed',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1)
