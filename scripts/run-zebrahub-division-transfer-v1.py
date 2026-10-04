"""Sequential frozen-encoder external-data transfer screen on Antelume."""
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
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def available(own=False):
    value=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout
    pids={int(x) for x in value.splitlines() if x.strip()}
    if pids-({os.getpid()} if own else set()):raise RuntimeError('Foreign GPU work exists; leave other projects untouched')


def execute(args,result):
    if sha(args.code/'CONTRACT.json')!=args.contract_sha256:raise ValueError('Code contract changed')
    contract=json.loads((args.code/'CONTRACT.json').read_text())
    for name,digest in contract['files'].items():
        if sha(args.code/name)!=digest:raise ValueError('Recipe or external data changed')
    if sha(args.base/'BUNDLE.json')!='467076cacc2587444267f6e14f4edb93dd578c27fb1eff76ffbc06cbb290bdb6':raise ValueError('Base changed')
    base=json.loads((args.base/'BUNDLE.json').read_text())
    for name,digest in base['files'].items():
        if sha(args.base/name)!=digest:raise ValueError('Base data/warm start changed')
    if args.mode=='full':
        smoke=json.loads(args.smoke.read_text())
        if smoke['status']!='functionality_passed' or smoke['contract_sha256']!=args.contract_sha256:raise ValueError('Exact-code smoke required')
    available()
    if shutil.disk_usage(args.output).free<128*1024**2:raise ValueError('Insufficient storage')
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path.insert(0,str(args.code))
    import numpy as np
    import torch
    from research.zebrahub_division_transfer import aligned_biohub,aligned_external,aligned_context,shared_geometry
    from research.zebrahub_transfer_head import fit_transfer
    from research.frozen_image_head import morphology,predict
    from research.image_context_quality import metrics
    from research.temporal_contrastive.graph_context_division_model import GraphContextDivisionModel,load_backbone_checkpoint
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one idle GPU required')
    torch.set_num_threads(2);torch.manual_seed(20260913);torch.cuda.set_per_process_memory_fraction(.45)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    external_manifest=json.loads((args.code/'EXTERNAL.json').read_text())
    if external_manifest['source']!='ZSNS004' or external_manifest['external_validation_opened']:raise ValueError('External exclusion failed')
    bio_manifest=json.loads((args.base/'data/MANIFEST.json').read_text())
    encoder_seconds=0.;head_seconds=0.;outcomes={}

    def load_encoder(target):
        model=GraphContextDivisionModel()
        warm=torch.load(args.base/f'warm_start_target_{target}.pt',map_location='cpu',weights_only=True)
        load_backbone_checkpoint(model,warm);model.backbone=model.backbone.to('cuda').eval().requires_grad_(False)
        return model

    def state_hash(model):
        h=hashlib.sha256()
        for k,v in sorted(model.backbone.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def encode(model,patches,geometry,context,mask):
        values=[]
        for start in range(0,len(patches),16):
            p=torch.from_numpy(patches[start:start+16]).to('cuda');g=torch.from_numpy(shared_geometry(geometry[start:start+16])).to('cuda')
            with torch.autocast('cuda',dtype=torch.float16):v=model.relational_features(p,g)
            values.append(v.float().cpu().numpy())
        return np.concatenate((np.concatenate(values),morphology(context,mask)),axis=1)

    def subset_smoke(data):
        rows=np.r_[np.flatnonzero(data['targets']>.5)[:2],np.flatnonzero(data['targets']<=.5)[:2]]
        if len(rows)!=4:raise ValueError('Real smoke requires both source classes')
        return {k:v[rows] for k,v in data.items()}

    def external_features(model,smoke=False):
        chunks=[]
        records=external_manifest['files'][:1] if smoke else external_manifest['files']
        for index,record in enumerate(records):
            available(own=True)
            with np.load(args.code/record['external_path'],allow_pickle=False) as a:raw=dict(a)
            with np.load(args.code/record['descriptor_path'],allow_pickle=False) as a:data=dict(a)
            if smoke:data=subset_smoke(data)
            p=aligned_external(raw['source_patches'],raw['target_patches'],data['indices'])
            x=encode(model,p,data['geometry'],data['context'],data['mask'])
            chunks.append(dict(features=x,targets=data['targets'],weights=data['weights'],eligible=data['eligible']))
            if (index+1)%16==0:print(json.dumps(dict(event='external_encoded',transitions=index+1,total=len(records))),flush=True)
        return {k:np.concatenate([p[k] for p in chunks]) for k in chunks[0]}

    def bio_features(model,embryo,role,smoke=False):
        available(own=True)
        row=bio_manifest['files'][f'{embryo}-{role}.npz']
        inventory=json.loads((args.base/'data'/row['inventory_path']).read_text())
        if any(x['embryo']!=embryo or x['role']!=role for x in inventory):raise ValueError('Source embryo/role mismatch')
        with np.load(args.base/'data'/f'{embryo}-{role}.npz',allow_pickle=False) as a:data=dict(a)
        if smoke:data=subset_smoke(data)
        c,m=aligned_context(data['context'],data['mask'])
        x=encode(model,aligned_biohub(data['patches']),data['geometry'],c,m)
        return dict(features=x,targets=data['targets'],weights=data['weights'],eligible=data['eligible'])

    for source in ('44b6','6bba')[:1 if args.mode=='smoke' else 2]:
        target='6bba' if source=='44b6' else '44b6';folder=args.output/f'source-{source}';folder.mkdir()
        tick=time.monotonic();model=load_encoder(target);before=state_hash(model)
        ext=external_features(model,args.mode=='smoke');bio=bio_features(model,source,'optimization',args.mode=='smoke')
        if args.mode=='smoke':
            warm=torch.load(args.base/f'warm_start_target_{target}.pt',map_location='cpu',weights_only=True)
            load_backbone_checkpoint(model,warm)
            replay=bio_features(model,source,'optimization',True)
            if not np.array_equal(bio['features'],replay['features']):raise ValueError('Encoder reload changed actual features')
            del warm
        else:selection=bio_features(model,source,'selection')
        if before!=state_hash(model):raise ValueError('Frozen encoder changed')
        torch.cuda.synchronize();encoder_seconds+=time.monotonic()-tick
        del model;gc.collect();torch.cuda.empty_cache()
        np.savez_compressed(folder/'external-features.npz',**ext);np.savez_compressed(folder/'optimization-features.npz',**bio)
        tick=time.monotonic();head=fit_transfer(ext,bio);np.savez_compressed(folder/'head.npz',**head)
        with np.load(folder/'head.npz',allow_pickle=False) as a:saved=dict(a)
        if not np.array_equal(predict(bio['features'],head),predict(bio['features'],saved)):raise ValueError('Saved-head replay changed')
        head_seconds+=time.monotonic()-tick
        if args.mode=='smoke':
            result.update(status='functionality_passed',encoder_reload_exact=True,saved_head_predictions_exact=True,
                external_rows=len(ext['targets']),biohub_rows=len(bio['targets']),head_parameters=len(head['coefficients'])+1)
        else:
            scores=predict(selection['features'],head);gate=selection['eligible'].astype(bool)
            row=metrics(selection['targets'][gate],scores[gate])
            outcome=dict(source_embryo=source,held_out_embryo=target,metrics=row,source_gate_passed=row['source_gate_passed'],
                head_sha256=sha(folder/'head.npz'),warm_start_sha256=sha(args.base/f'warm_start_target_{target}.pt'),penalty=.01)
            np.savez_compressed(folder/'selection-features.npz',**selection);np.savez_compressed(folder/'source-selection.npz',scores=scores,targets=selection['targets'],eligible=gate)
            write_json(folder/'terminal.json',outcome);outcomes[source]=outcome
            print(json.dumps(dict(event='source_head',source=source,**row)),flush=True)
            del selection,scores
        del ext,bio,head,saved;gc.collect()
    result['source_models']=outcomes
    if args.mode=='full':
        if not all(x['source_gate_passed'] for x in outcomes.values()):result['status']='rejected_source_selection'
        else:
            write_json(args.output/'frozen-source-policy.json',outcomes);result['held_out_embryo_scores_opened']=True
            held_out={}
            for source,outcome in outcomes.items():
                target=outcome['held_out_embryo'];tick=time.monotonic();model=load_encoder(target)
                parts=[bio_features(model,target,role) for role in ('optimization','selection')]
                torch.cuda.synchronize();encoder_seconds+=time.monotonic()-tick
                del model;gc.collect();torch.cuda.empty_cache()
                data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
                with np.load(args.output/f'source-{source}/head.npz',allow_pickle=False) as a:head=dict(a)
                scores=predict(data['features'],head);gate=data['eligible'].astype(bool)
                row=metrics(data['targets'][gate],scores[gate],threshold=outcome['metrics']['source_threshold']);held_out[target]=row
                np.savez_compressed(args.output/f'held-out-{target}.npz',**data,scores=scores)
            counts=[r['fixed_threshold'] for r in held_out.values()]
            passed=(all(r['average_precision']>=.55 and r['fixed_threshold']['tp']>=1 for r in held_out.values())
                    and sum(c['tp'] for c in counts)>=3 and sum(c['fp'] for c in counts)<=1)
            result.update(status='passed_patch_pilot' if passed else 'rejected_embryo_transfer',held_out=held_out,patch_pilot_gate_passed=passed)
    result.update(encoder_stage_seconds=encoder_seconds,head_stage_seconds=head_seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated())


def main():
    p=argparse.ArgumentParser()
    for name in ('code','base','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--contract-sha256',required=True);p.add_argument('--mode',choices=('smoke','full'),required=True);p.add_argument('--smoke',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(run_id='zebrahub-division-transfer-v1',mode=args.mode,status='running',contract_sha256=args.contract_sha256,
        authorized_for_submission=False,held_out_embryo_scores_opened=False,competition_test_data_read=False,
        external_validation_opened=False,kaggle_gpu_hours=0,backbone_trainable=False,public_predictions_used=False)
    def timeout():write_json(args.output/'timeout.json',dict(status='walltime_limit',partial_artifacts_preserved=True));os._exit(124)
    timer=threading.Timer(120 if args.mode=='smoke' else 900,timeout);timer.daemon=True;timer.start()
    try:execute(args,result)
    except BaseException as error:result.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        timer.cancel();result['elapsed_seconds']=time.monotonic()-start;write_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
