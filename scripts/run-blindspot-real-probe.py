"""Bounded100-step training-only image-restoration check; no annotation IO."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main(args):
    import numpy as np
    import torch
    import zarr
    from blindspot_restoration import build_model
    from blindspot_training_contract import identity,origin,PATCH,FRAMES,proxy_gate
    started = time.monotonic()
    policy = identity(args.manifest.read_bytes())
    cpu = json.loads((args.runtime/'cpu_probe.json').read_text())
    model_source = args.runtime/'blindspot_restoration.py'
    if (cpu['status'] != 'passed_synthetic_blindspot_functionality'
            or hashlib.sha256(model_source.read_bytes()).hexdigest()!=cpu['source_sha256']['blindspot_restoration.py']):
        raise ValueError('Exact CPU-tested model implementation required')
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise ValueError('Exactly two T4s required')
    torch.set_num_threads(2)
    torch.manual_seed(policy['seed']); torch.cuda.manual_seed_all(policy['seed'])
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    fitting,diagnostic,inventory = [],{},[]
    for stem in policy['fitting_stems']+policy['diagnostic_stems']:
        array = zarr.open_group(str(args.data/(stem+'.zarr')),mode='r')['0']
        if tuple(array.shape)!=(100,64,256,256):
            raise ValueError('Expected complete native training image')
        patches=[]
        for frame in FRAMES:
            start=origin(stem,frame)
            patch=np.asarray(array[(frame,)+tuple(slice(s,s+n) for s,n in zip(start,PATCH))],dtype=np.float32)
            if patch.shape!=PATCH or not np.isfinite(patch).all():
                raise ValueError('Incomplete/nonfinite native image patch')
            patches.append(patch)
            inventory.append(dict(stem=stem,frame=frame,origin=start,shape=PATCH,
                sha256=hashlib.sha256(patch.tobytes()).hexdigest(),scope='fit' if stem in policy['fitting_stems'] else 'diagnostic'))
        if stem in policy['fitting_stems']: fitting.extend(patches)
        else: diagnostic[stem]=np.stack(patches)
        if len(inventory)%60==0:
            print(json.dumps(dict(images_loaded=len(inventory),ground_truth_opened=False)),flush=True)
    training=np.stack(fitting)
    low,high=map(float,np.quantile(training,[.001,.999]))
    if not np.isfinite([low,high]).all() or high<=low:
        raise ValueError('Nondegenerate fitting-only normalization required')
    scale=high-low
    training=torch.from_numpy((training-low)/scale)[:,None]
    diagnostic={s:torch.from_numpy((v-low)/scale)[:,None] for s,v in diagnostic.items()}
    policy.update(normalization_low=low,normalization_high=high,normalization_fit_patches=288,
                  diagnostic_patches=72,model_source_sha256=cpu['source_sha256']['blindspot_restoration.py'])
    (args.output/'patch_inventory.json').write_text(json.dumps(inventory,indent=2))
    (args.output/'identity.json').write_text(json.dumps(policy,indent=2))
    core=build_model(16).cuda()
    if sum(p.numel() for p in core.parameters())!=14321:
        raise ValueError('Unexpected model architecture')
    model=torch.nn.DataParallel(core)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    rng=torch.Generator().manual_seed(policy['seed'])
    crop=(slice(None),slice(None),slice(7,-7),slice(7,-7),slice(7,-7))
    losses=[]
    for step in range(100):
        batch=training[torch.randint(len(training),(4,),generator=rng)].cuda()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda',dtype=torch.float16):
            output=model(batch)
            loss=torch.nn.functional.mse_loss(output[crop].float(),batch[crop])
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        if torch.count_nonzero(core.first.weight.grad[:,:,1,1,1]):
            raise ValueError('Masked center received training gradient')
        torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True)
        old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()<old_scale: raise ValueError('Optimizer step skipped; do not silently extend run')
        losses.append(float(loss.detach()))
        if (step+1)%20==0: print(json.dumps(dict(step=step+1,loss=losses[-1])),flush=True)
    model.eval()
    rows=[]
    kernel=torch.ones(1,1,3,3,3,device='cuda')/26
    kernel[:,:,1,1,1]=0
    with torch.no_grad():
        for stem in policy['diagnostic_stems']:
            batch=diagnostic[stem].cuda()
            with torch.amp.autocast('cuda',dtype=torch.float16): prediction=model(batch)
            control=torch.nn.functional.conv3d(batch,kernel,padding=1)
            if not torch.isfinite(prediction).all(): raise ValueError('Nonfinite diagnostic output')
            error=(prediction[crop].float()-batch[crop]).double()
            reference=(control[crop]-batch[crop]).double()
            rows.append(dict(stem=stem,pixels=error.numel(),model_sse=float(error.square().sum()),
                             neighbor_sse=float(reference.square().sum()),prediction_std=float(prediction[crop].float().std())))
    if max(r['prediction_std'] for r in rows)<=0: raise ValueError('Collapsed constant restoration')
    sample=training[:1,:,:17,:18,:19].cuda().requires_grad_(True)
    gradient,=torch.autograd.grad(core(sample)[0,0,8,9,9],sample)
    if float(gradient[0,0,8,9,9])!=0 or float(gradient.abs().sum())<=0:
        raise ValueError('Real trained model failed nontrivial blind spot check')
    checkpoint=args.output/'last.pt'
    torch.save(dict(model={k:v.detach().cpu() for k,v in core.state_dict().items()},identity=policy,step=100),checkpoint)
    restored=build_model(16).cuda().eval()
    restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    with torch.no_grad():
        if not torch.equal(core(sample.detach()),restored(sample.detach())):
            raise ValueError('Exact final checkpoint replay failed')
    result=dict(status='completed_real_blindspot_probe',identity=policy,steps=100,losses=losses,
        per_movie=rows,comparison=proxy_gate(rows,policy['diagnostic_stems']),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_replay_exact=True,
        trained_center_gradient=0.,trained_other_gradient_l1=float(gradient.abs().sum()),
        inventory_sha256=hashlib.sha256((args.output/'patch_inventory.json').read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,ground_truth_opened=False,authorized_for_submission=False)
    (args.output/'probe_result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result['comparison']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('repo','runtime','manifest','data','output'): parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--steps',type=int,choices=[100],required=True)
    main(parser.parse_args())
