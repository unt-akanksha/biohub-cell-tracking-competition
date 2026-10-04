"""Training-only frozen-network calibration cache and bounded convex fit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

CHECKPOINT_SHA = '76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPLIT_SHA = '12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def scope(split,profile):
    fold = split['folds'][0]
    diagnostic = fold['train'][::5]
    fitting = [s for s in fold['train'] if s not in diagnostic]
    if (len(fitting)!=96 or len(diagnostic)!=24 or len(set(fold['train']))!=120
        or set(fold['train'])&set(fold['selection']+fold['audit_order'])):
        raise ValueError('Original training-only movie partition required')
    if profile not in ('probe','full'):
        raise ValueError('Unknown scope')
    return (fitting[:4],diagnostic[:2]) if profile=='probe' else (fitting,diagnostic)


def main(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path[:0] = [str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import train_unet_transformer as official
    from independent_real_baseline import install_empty_attention_guard
    from image_motion_residual import embedded_flow,flow_hash
    from backward_flow_ops import sample_backward_flow
    from independent_motion_prior import SCALE,VARIANCE
    from annotated_missing_parent import batch_missing_parent_masks
    from association_calibration import pack_sample,combine,fit,metrics,FLOW_CONTROL,NEURAL_CONTROL
    from seeded_frame_dataset import seed_dataset
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise RuntimeError('Bounded two-T4 profile required')
    torch.set_num_threads(2); torch.manual_seed(20260910); torch.cuda.manual_seed_all(20260910)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA:
        raise ValueError('Exact completed integrated checkpoint required')
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=SPLIT_SHA:
        raise ValueError('Frozen split changed')
    split = json.loads(args.manifest.read_text())
    fitting,diagnostic = scope(split,args.scope)
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (state['step']!=1000 or state['identity']['training_stems']!=split['folds'][0]['train']
        or state['identity']['max_steps']!=1000):
        raise ValueError('Completed training-only checkpoint required')
    probe = None
    if args.scope=='full':
        probe = json.loads((args.runtime/'probe_result.json').read_text())
        if (probe['profile']!='probe' or not probe['functionality_passed']
            or probe['checkpoint_sha256']!=CHECKPOINT_SHA or probe['split_sha256']!=SPLIT_SHA):
            raise ValueError('Verified small real-image probe required')
    model = official.UNetNodeTransformer(official.TemporalUNet3D(
        in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda(0)
    model.load_state_dict(state['model'],strict=True)
    model.requires_grad_(False).eval()
    install_empty_attention_guard(model)
    flow = embedded_flow(state,'cuda:1')
    before = dict(neural=flow_hash(model),flow=flow_hash(flow))
    del state
    args.output.mkdir(parents=True,exist_ok=True)
    records,packed = [],dict(fitting=[],diagnostic=[])
    started = time.monotonic()
    with torch.no_grad():
        for role,stems in [('fitting',fitting),('diagnostic',diagnostic)]:
            for stem in stems:
                vm,windows = official.load_dataset_windows(args.data/stem,window_size=2,downsample=(1,4,4))
                observable = [w for w in windows if bool(w.targets[0].sum()>0)]
                chosen = sorted(observable,key=lambda w:hashlib.sha256(
                    f'calibration-v1:{stem}:{w.t_start}'.encode()).hexdigest())[:3]
                if len(chosen)!=3:
                    raise ValueError('Three annotated windows required: '+stem)
                dataset = official.FrameWindowDataset([(vm,chosen)],augmentations=[])
                seed_dataset(official,dataset,20260910)
                for window,batch in zip(chosen,DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)):
                    imgs = batch['imgs'].cuda(0).float()
                    coords,masks,targets = (batch[k].cuda(0) for k in ('coords','masks','targets'))
                    ds = batch['downsample'][0].cuda(0)
                    with torch.amp.autocast('cuda',dtype=torch.float16):
                        encoded,logits = model.encode(imgs)
                        field = flow(imgs.to('cuda:1').half().float()).to('cuda:0')
                    encoded,logits = encoded.float(),[a.float() for a in logits]
                    frames = []
                    for i in range(2):
                        dc,dp,dm,matches = official.detect_and_match(logits[i],coords[:,i],masks[:,i],
                            tuple(batch['image_shape'][0].tolist()),voxel_size=tuple(batch['voxel_size'][0].tolist()),
                            pool_kernel_um=5.,frame_index=i,window_size=2)
                        if int(dm.sum())>2048:
                            raise ValueError('Attention limit exceeded; no truncation')
                        frames.append((dc,dp,dm,matches,model._index_features(encoded[:,i],dc,dm)))
                    source,target = frames
                    ns,nt = int(source[2].sum()),int(target[2].sum())
                    truth = official.build_matched_edge_targets(source[3],target[3],targets[:,0],
                        source[0].shape[1],target[0].shape[1])
                    neural = model.predict_edges(source[4],target[4],source[0]*ds,target[0]*ds,
                        source[1],target[1],source[2],target[2])
                    displacement,valid = sample_backward_flow(field,target[0])
                    if (target[2]&~valid).any():
                        raise ValueError('Real target outside flow grid')
                    delta = (source[0].unsqueeze(-2)-target[0].unsqueeze(-3))*ds*imgs.new_tensor(SCALE)
                    prior = -.5*((delta-displacement.unsqueeze(-3)).square()/imgs.new_tensor(VARIANCE)).sum(-1)
                    nulls = batch_missing_parent_masks(source,target,targets[:,0],coords[:,0]*ds,ds)[0]
                    sample = pack_sample(neural[0,:ns,:nt].cpu().numpy(),prior[0,:ns,:nt].cpu().numpy(),
                        truth[0,:ns,:nt].cpu().numpy(),nulls)
                    path = args.output/f'{stem}_{window.t_start}.npz'
                    np.savez_compressed(path,**sample)
                    with np.load(path,allow_pickle=False) as loaded:
                        if any(not np.array_equal(loaded[k],sample[k]) for k in sample):
                            raise ValueError('Cache reload changed features or labels')
                    packed[role].append(sample)
                    records.append(dict(role=role,stem=stem,t_start=window.t_start,source_nodes=ns,target_nodes=nt,
                        supervised_columns=len(sample['starts']),known_null_columns=int(np.sum(nulls)),
                        input_sha256=dataset.sample_hashes[-1],cache_file=path.name,
                        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                print(json.dumps(dict(role=role,stem=stem,windows=len(chosen),elapsed_seconds=time.monotonic()-started)),flush=True)
    after = dict(neural=flow_hash(model),flow=flow_hash(flow))
    if before!=after:
        raise ValueError('Frozen network state changed')
    replayed = False
    if probe is not None:
        lookup = {(r['stem'],r['t_start']):r for r in records}
        for old in probe['records']:
            now = lookup[(old['stem'],old['t_start'])]
            if old['input_sha256']!=now['input_sha256'] or old['cache_sha256']!=now['cache_sha256']:
                raise ValueError('Probe input/features did not replay')
        replayed = True
    train_data,diagnostic_data = (combine(packed[k]) for k in ('fitting','diagnostic'))
    fitted = fit(train_data)
    if args.scope=='full' and fitted['fit_metrics']['known_null_columns']==0:
        raise ValueError('Full calibration needs observed missing-parent examples')
    result = dict(status='completed_calibration_not_tracking_validation',profile=args.scope,
        checkpoint_sha256=CHECKPOINT_SHA,split_sha256=SPLIT_SHA,fitting_stems=fitting,diagnostic_stems=diagnostic,
        frozen_before=before,frozen_after=after,records=records,fit=fitted,
        diagnostic=dict(calibrated=metrics(fitted['parameters'],diagnostic_data),
            flow=metrics(FLOW_CONTROL,diagnostic_data),neural=metrics(NEURAL_CONTROL,diagnostic_data)),
        functionality_passed=True,probe_inputs_replayed=replayed,elapsed_seconds=time.monotonic()-started,
        diagnostic_scope='Calibration-held-out windows within original neural training pool; not independent model validation',
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','fit','diagnostic','functionality_passed')}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','manifest','checkpoint','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scope',choices=('probe','full'),default='probe')
    main(parser.parse_args())
