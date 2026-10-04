"""Small Antelume-only frozen-feature screen; source calibration precedes transfer."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def write_json(path, data):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def available(own=False):
    value=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=15).stdout
    pids={int(x.strip()) for x in value.splitlines() if x.strip()}
    if pids-({os.getpid()} if own else set()):
        raise RuntimeError('Another project is using the GPU; leave it untouched')


def execute(args,result):
    if sha(args.bundle/'BUNDLE.json') != '467076cacc2587444267f6e14f4edb93dd578c27fb1eff76ffbc06cbb290bdb6':
        raise ValueError('Base data/runtime contract changed')
    base=json.loads((args.bundle/'BUNDLE.json').read_text())
    for name,digest in base['files'].items():
        if sha(args.bundle/name) != digest: raise ValueError('Base input changed')
    if sha(args.code/'CONTRACT.json') != args.contract_sha256:
        raise ValueError('New source contract changed')
    contract=json.loads((args.code/'CONTRACT.json').read_text())
    for name,digest in contract['files'].items():
        if sha(args.code/name) != digest: raise ValueError('Frozen source changed')
    if args.mode=='full':
        smoke=json.loads(args.smoke.read_text())
        if smoke['status']!='functionality_passed' or smoke['contract_sha256']!=args.contract_sha256:
            raise ValueError('Require exact-code successful smoke')
    if shutil.disk_usage(args.output).free < 128*1024**2:
        raise ValueError('Insufficient storage headroom')
    available()
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path[:0]=[str(args.code),str(args.bundle)]
    import numpy as np
    import torch
    from frozen_image_head import morphology,fit,predict,PENALTIES
    from research.image_context_quality import metrics,utility
    from research.temporal_contrastive.graph_context_division_model import GraphContextDivisionModel,load_backbone_checkpoint
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one freed Antelume GPU required')
    torch.set_num_threads(2);torch.manual_seed(20260913)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.45)
    manifest=json.loads((args.bundle/'data/MANIFEST.json').read_text())
    source_models={}; gpu_seconds=0.; head_seconds=0.

    def packet(embryo,role):
        name=f'{embryo}-{role}.npz'; row=manifest['files'][name]
        if sha(args.bundle/'data'/name)!=row['sha256']:raise ValueError('Data changed')
        inventory=json.loads((args.bundle/'data'/row['inventory_path']).read_text())
        if any(x['embryo']!=embryo or x['role']!=role for x in inventory):
            raise ValueError('Role or embryo mismatch')
        with np.load(args.bundle/'data'/name,allow_pickle=False) as a: data=dict(a)
        return data

    def encoder(target):
        model=GraphContextDivisionModel()
        state=torch.load(args.bundle/f'warm_start_target_{target}.pt',map_location='cpu',weights_only=True)
        load_backbone_checkpoint(model,state)
        model.backbone=model.backbone.to('cuda').eval().requires_grad_(False)
        return model

    def backbone_hash(model):
        h=hashlib.sha256()
        for k,v in sorted(model.backbone.state_dict().items()):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def extract(model,data):
        output=[]
        for start in range(0,len(data['patches']),16):
            if start%160==0: available(own=True)
            patches=torch.from_numpy(data['patches'][start:start+16]).to('cuda')
            geometry=torch.from_numpy(data['geometry'][start:start+16]).to('cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                features=model.relational_features(patches,geometry)
            output.append(features.float().cpu().numpy())
        return np.concatenate((np.concatenate(output),morphology(data['context'],data['mask'])),axis=1)

    def fit_source(source,train,select,x_train,x_select,folder):
        best=None; history=[]
        for penalty in PENALTIES:
            tick=time.monotonic()
            try:
                state=fit(x_train,train['targets'],train['eligible'],train['weights'],penalty)
            except RuntimeError as error:
                history.append(dict(penalty=penalty,status='unconverged',message=str(error)))
                continue
            values=predict(x_select,state)
            eligible=select['eligible'].astype(bool)
            row=metrics(select['targets'][eligible],values[eligible])
            history.append(dict(penalty=penalty,status='converged',iterations=state['iterations'],
                                metrics=row,seconds=time.monotonic()-tick))
            if best is None or utility(row)>utility(best['metrics']):
                best=dict(penalty=penalty,metrics=row)
                np.savez_compressed(folder/'best-head.npz',**state)
                np.savez_compressed(folder/'source-selection.npz',scores=values,
                                    targets=select['targets'],eligible=eligible)
            print(json.dumps(dict(event='source_head',source=source,penalty=penalty,**row)),flush=True)
        write_json(folder/'selection-history.json',dict(rows=history))
        if best is None: raise RuntimeError('No convex fit converged')
        best.update(source_embryo=source,held_out_embryo='6bba' if source=='44b6' else '44b6',
                    head_sha256=sha(folder/'best-head.npz'),source_gate_passed=best['metrics']['source_gate_passed'])
        write_json(folder/'terminal.json',best)
        return best

    for source in ('44b6','6bba')[:1 if args.mode=='smoke' else 2]:
        target='6bba' if source=='44b6' else '44b6'
        folder=args.output/f'source-{source}';folder.mkdir()
        train=packet(source,'optimization')
        if args.mode=='smoke':
            positive=np.flatnonzero(train['targets']>.5)[:4]
            negative=np.flatnonzero(train['targets']<=.5)[:4]
            rows=np.r_[positive,negative]
            train={k:v[rows] for k,v in train.items()}
        tick=time.monotonic();model=encoder(target);before=backbone_hash(model)
        x_train=extract(model,train)
        if args.mode=='smoke':
            # Exact weight reload and repeated physical patch encoding, not synthetic features.
            state=torch.load(args.bundle/f'warm_start_target_{target}.pt',map_location='cpu',weights_only=True)
            load_backbone_checkpoint(model,state)
            replay=extract(model,train)
            if not np.array_equal(x_train,replay):raise ValueError('Encoder reload changed features')
            del state
        else:
            select=packet(source,'selection');x_select=extract(model,select)
        if before!=backbone_hash(model):raise ValueError('Frozen encoder state changed')
        torch.cuda.synchronize();gpu_seconds+=time.monotonic()-tick
        del model;gc.collect();torch.cuda.empty_cache()
        np.savez_compressed(folder/'optimization-features.npz',features=x_train,
            targets=train['targets'],eligible=train['eligible'],weights=train['weights'])
        tick=time.monotonic()
        if args.mode=='smoke':
            head=fit(x_train,train['targets'],train['eligible'],train['weights'],.01)
            scores=predict(x_train,head);np.savez_compressed(folder/'smoke-head.npz',**head)
            with np.load(folder/'smoke-head.npz',allow_pickle=False) as a:replay=predict(x_train,dict(a))
            if not np.array_equal(scores,replay):raise ValueError('Saved head replay mismatch')
            result.update(status='functionality_passed',encoder_reload_exact=True,
                          saved_head_predictions_exact=True,smoke_rows=len(x_train),
                          head_parameters=len(head['coefficients'])+1)
        else:
            np.savez_compressed(folder/'selection-features.npz',features=x_select,
                targets=select['targets'],eligible=select['eligible'],weights=select['weights'])
            source_models[source]=fit_source(source,train,select,x_train,x_select,folder)
            del select,x_select
        head_seconds+=time.monotonic()-tick
        del train,x_train;gc.collect()
    result['source_models']=source_models
    if args.mode=='full':
        if not all(x['source_gate_passed'] for x in source_models.values()):
            result.update(status='rejected_source_selection')
        else:
            write_json(args.output/'frozen-source-policy.json',source_models)
            result['held_out_embryo_scores_opened']=True
            held_out={}
            for source,outcome in source_models.items():
                target=outcome['held_out_embryo'];tick=time.monotonic()
                model=encoder(target);before=backbone_hash(model)
                features=[];labels=[];gates=[]
                for role in ('optimization','selection'):
                    data=packet(target,role);features.append(extract(model,data))
                    labels.append(data['targets']);gates.append(data['eligible']);del data
                if before!=backbone_hash(model):raise ValueError('Target encoding changed backbone')
                torch.cuda.synchronize();gpu_seconds+=time.monotonic()-tick
                del model;gc.collect();torch.cuda.empty_cache()
                with np.load(args.output/f'source-{source}/best-head.npz',allow_pickle=False) as a:head=dict(a)
                scores=predict(np.concatenate(features),head);y=np.concatenate(labels);gate=np.concatenate(gates).astype(bool)
                row=metrics(y[gate],scores[gate],threshold=outcome['metrics']['source_threshold'])
                held_out[target]=row
                np.savez_compressed(args.output/f'held-out-{target}.npz',features=np.concatenate(features),
                                    scores=scores,targets=y,eligible=gate)
            counts=[x['fixed_threshold'] for x in held_out.values()]
            passed=(all(x['average_precision']>=.55 and x['fixed_threshold']['tp']>=1 for x in held_out.values())
                    and sum(x['tp'] for x in counts)>=3 and sum(x['fp'] for x in counts)<=1)
            result.update(status='passed_patch_pilot' if passed else 'rejected_embryo_transfer',
                          held_out=held_out,held_out_embryo_scores_opened=True,patch_pilot_gate_passed=passed)
    result.update(encoder_stage_seconds=gpu_seconds,head_stage_seconds=head_seconds,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated())


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','code','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--mode',choices=('smoke','full'),required=True)
    parser.add_argument('--smoke',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result=dict(run_id='frozen-image-head-v1',mode=args.mode,status='running',
        contract_sha256=args.contract_sha256,authorized_for_submission=False,kaggle_gpu_hours=0,
        competition_test_data_read=False,held_out_embryo_scores_opened=False,
        backbone_trainable=False,public_predictions_used=False)
    start=time.monotonic()
    def timeout():
        write_json(args.output/'timeout.json',dict(status='walltime_limit',partial_artifacts_preserved=True));os._exit(124)
    timer=threading.Timer(120 if args.mode=='smoke' else 900,timeout);timer.daemon=True;timer.start()
    try:execute(args,result)
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        timer.cancel();result['elapsed_seconds']=time.monotonic()-start
        write_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
