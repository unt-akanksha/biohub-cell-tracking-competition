"""Ten real detector-only updates with owned, frozen view-consensus targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPLIT_SHA='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
RAW_PROBE_SHA='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba'


def training_profile(profile,objective,logit_targets,probe_bytes=None):
    if profile=='probe' and objective=='pu':
        return 10,None
    if profile!='full' or objective not in ('pu','sparse') or not logit_targets or probe_bytes is None:
        raise ValueError('Only verified raw-target full fits or the small PU probe are allowed')
    if hashlib.sha256(probe_bytes).hexdigest()!=RAW_PROBE_SHA:
        raise ValueError('Exact completed raw-target probe required')
    probe=json.loads(probe_bytes)
    if (probe['status']!='passed_owned_detector_pu_functionality' or probe['steps']!=10
        or probe['identity']['teacher_peak_order']!='raw_logits' or not probe['strict_reload_identical']
        or any(not probe[k] for k in ('teacher_unchanged','linker_unchanged','bn_unchanged','detector_changed'))):
        raise ValueError('Successful real raw-target functionality required')
    return 1000,probe


def scope(identity,split):
    fold=split['folds'][0]
    if identity['training_stems']!=fold['train'] or len(fold['train'])!=120:
        raise ValueError('Exact source-only parent training membership required')
    if set(fold['train'])&set(fold['selection']+fold['audit_order']):
        raise ValueError('Parent/source/audit overlap')
    stems=fold['train'][:4]
    if any(not s.startswith('6bba_') for s in stems):
        raise ValueError('Only four recorded source-training movies may enter probe')
    return stems


def main(args):
    if (hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA
        or hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=SPLIT_SHA):
        raise ValueError('Exact independent parent and original split required')
    sys.path[:0]=[str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_unet_transformer as official
    from owned_detector_pu import contract,targets,loss,teacher_probabilities
    logit_targets=getattr(args,'logit_targets',False)
    if logit_targets:
        from owned_detector_logit_targets import contract,targets
    profile=getattr(args,'fit_profile','probe'); objective_name=getattr(args,'objective','pu')
    steps,probe=training_profile(profile,objective_name,logit_targets,
        (args.runtime/'raw_probe_result.json').read_bytes() if profile=='full' else None)
    from seeded_frame_dataset import seed_dataset
    from independent_real_baseline import fresh_batches
    from image_motion_residual import flow_hash
    from real_checkpoint_gpu_smoke import run_smoke
    from detector_spatial_tta import install_detector_spatial_tta
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise RuntimeError('Two CUDA devices required for bounded detector probe')
    torch.set_num_threads(2)
    seed=20260910
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    stems=scope(state['identity'],json.loads(args.manifest.read_text()))
    if state['step']!=1000 or not state['identity'].get('image_motion'):
        raise ValueError('Completed frozen image-model parent required')
    args.output.mkdir(parents=True,exist_ok=True)

    def model():
        result=official.UNetNodeTransformer(official.TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128]),
            unet_out_channels=32,pos_feat_dim=32).cuda()
        result.load_state_dict(state['model'],strict=True)
        return result.eval()

    teacher=model().requires_grad_(False)
    student=model(); student.transformer.requires_grad_(False)
    teacher_before=flow_hash(teacher); linker_before=flow_hash(student.transformer)
    detector_before=flow_hash(student.unet)+flow_hash(student.detect_head)
    bn_before={k:v.clone() for k,v in student.state_dict().items() if any(x in k for x in ('running_mean','running_var','num_batches_tracked'))}
    # Evaluation-mode BN/dropout, but gradients enabled for detector weights.
    # This isolates target/loss changes and avoids sparse small-batch BN drift.
    student.unet=torch.nn.DataParallel(student.unet)
    parameters=list(student.unet.parameters())+list(student.detect_head.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    videos=[official.load_dataset_windows(args.data/s,window_size=2,downsample=(1,4,4)) for s in stems]
    if any(not windows or not np.allclose(vm.voxel_size,[1.625]*3) for vm,windows in videos):
        raise ValueError('Nonempty source windows on the frozen physical grid required')
    dataset=seed_dataset(official,official.FrameWindowDataset(videos,augmentations=official.DEFAULT_AUGMENTATIONS),seed)
    generator=torch.Generator().manual_seed(seed)
    stream=fresh_batches(DataLoader(dataset,batch_size=2,shuffle=True,num_workers=0,generator=generator))
    full_dataset=None; full_generator=None; full_stream=None
    if profile=='full':
        full_videos=[official.load_dataset_windows(args.data/s,window_size=2,downsample=(1,4,4)) for s in state['identity']['training_stems']]
        if any(not windows or not np.allclose(vm.voxel_size,[1.625]*3) for vm,windows in full_videos):
            raise ValueError('Complete source training pool on the frozen grid required')
        full_dataset=seed_dataset(official,official.FrameWindowDataset(full_videos,augmentations=official.DEFAULT_AUGMENTATIONS),seed+1)
        full_generator=torch.Generator().manual_seed(seed+1)
        full_stream=fresh_batches(DataLoader(full_dataset,batch_size=2,shuffle=True,num_workers=0,generator=full_generator))
    identity=dict(state['identity'],run_id='owned-detector-pu-probe-v1',max_steps=steps,
        parent_identity=state['identity'],initialization_sha256=CHECKPOINT_SHA,owned_detector_pu=contract(),
        fine_tuning_stems=state['identity']['training_stems'] if profile=='full' else stems,
        fine_tuning_profile=profile,owned_detector_objective=objective_name,
        frozen_modules=['transformer','flow','batchnorm_statistics'],
        loss='Original sparse detector BCE control' if objective_name=='sparse' else 'Owned frozen view-consensus positive/unlabeled detector BCE',seed=seed,
        teacher_probability_precision='float32_before_sigmoid',teacher_peak_order='raw_logits' if logit_targets else 'probabilities',
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    history=[]; target_hashes=[]; started=time.monotonic(); first10_gate=None

    def attention_guard(model):
        original=model.predict_edges
        def bounded(*values):
            if max(values[0].shape[1],values[1].shape[1])>2048:
                raise RuntimeError('Too many smoke detections; abort instead of truncation')
            return original(*values)
        model.predict_edges=bounded
        return dict(maximum_nodes_per_frame=2048,action='abort_without_truncation')

    def input_hashes():
        return dataset.sample_hashes+([] if full_dataset is None else full_dataset.sample_hashes)

    def save(step):
        payload=dict(model={k.replace('unet.module.','unet.',1):v for k,v in student.state_dict().items()},
            frozen_flow_model=state['frozen_flow_model'],identity=identity,step=step,
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),input_hashes=input_hashes(),target_hashes=target_hashes,
            loader_rng=generator.get_state(),augmentation_rng=dataset.augmentation_rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if full_dataset is not None:
            payload.update(full_loader_rng=full_generator.get_state(),full_augmentation_rng=full_dataset.augmentation_rng.bit_generator.state)
        torch.save(payload,args.output/'last.tmp.pt'); os.replace(args.output/'last.tmp.pt',args.output/'last.pt')

    for step in range(1,steps+1):
        batch=next(full_stream if step>10 and full_stream is not None else stream); images=batch['imgs'].cuda().float()
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            _,native=teacher.encode(images)
            _,reflected=teacher.encode(images.flip(-1))
        labels=[]; counts=dict(consensus=0,annotations=0,boundary_annotations=0,positive_voxels=0,unknown_voxels=0,background_voxels=0,
            legacy_unit_probability_voxels=0,fp32_unit_probability_voxels=0)
        for w in range(2):
            for b in range(images.shape[0]):
                points=batch['coords'][b,w][batch['masks'][b,w]].numpy()
                inside=((points>=0)&(points<=np.asarray(images.shape[2:])-1)).all(-1)
                counts['boundary_annotations']+=int((~inside).sum())
                native_prob=teacher_probabilities(native[w][b,0])
                aligned_prob=teacher_probabilities(reflected[w][b,0].flip(-1))
                counts['legacy_unit_probability_voxels']+=int((native[w][b,0].sigmoid()==1).sum())
                counts['fp32_unit_probability_voxels']+=int((native_prob==1).sum())
                target=targets(native[w][b,0].float().cpu().numpy(),reflected[w][b,0].flip(-1).float().cpu().numpy(),points[inside]) if logit_targets else targets(native_prob.cpu().numpy(),aligned_prob.cpu().numpy(),points[inside])
                labels.append((w,b,target))
                digest=hashlib.sha256()
                for key in ('heatmap','weights','positive_coords'):
                    digest.update(key.encode()+b'\0'+target[key].tobytes())
                target_hashes.append(digest.hexdigest())
                counts['consensus']+=target['consensus_count']; counts['annotations']+=target['forced_annotation_count']
                for key,mask in [('positive_voxels','positive_mask'),('unknown_voxels','unknown_mask'),('background_voxels','background_mask')]:
                    counts[key]+=int(target[mask].sum())
        if counts['annotations']<=0 or counts['positive_voxels']<=0 or counts['unknown_voxels']<=0:
            raise ValueError('Real annotated, positive and uncertain training regions required')
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            _,predictions=student.encode(images)
        if objective_name=='sparse':
            objective=torch.stack([official.compute_detection_loss(predictions[w].float(),batch['coords'][:,w].cuda(),
                batch['masks'][:,w].cuda(),neg_weight=.01) for w in range(2)]).mean()
        else:
            objective=torch.stack([loss(predictions[w][b,0].float(),torch.from_numpy(target['heatmap']).cuda(),
                torch.from_numpy(target['weights']).cuda()) for w,b,target in labels]).mean()
        if not torch.isfinite(objective): raise ValueError('Nonfinite real PU objective')
        scaler.scale(objective).backward(); scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        row=dict(step=step,loss=float(objective.detach()),grad_norm=float(norm),elapsed_seconds=time.monotonic()-started,**counts)
        history.append(row)
        if step<=10 or step%25==0: print(json.dumps(row),flush=True)
        if step==10 and probe is not None:
            if input_hashes()!=probe['input_hashes'] or target_hashes!=probe['target_hashes']:
                raise ValueError('Verified raw-target probe inputs/targets were not replayed')
            print(json.dumps(dict(probe_replay_passed=True,objective=objective_name)),flush=True)
        if step==10 or step%50==0: save(step)
        if step==10 and profile=='full':
            first10_gate=run_smoke(args.output/'last.pt',args.data/stems[0],args.output/'first10_gate',
                standalone_image_flow=True,pre_motion_patch=install_detector_spatial_tta,encode_patch=attention_guard)
            if first10_gate['status']!='passed' or first10_gate['predicted_nodes']<=0:
                raise ValueError('First ten updates must pass graph functionality before extension')
            print(json.dumps(dict(first10_graph_gate_passed=True,objective=objective_name)),flush=True)
    student.unet=student.unet.module
    if flow_hash(teacher)!=teacher_before or flow_hash(student.transformer)!=linker_before:
        raise ValueError('Frozen teacher/linker changed')
    if any(not torch.equal(v,student.state_dict()[k]) for k,v in bn_before.items()):
        raise ValueError('Frozen normalization statistics changed')
    if flow_hash(student.unet)+flow_hash(student.detect_head)==detector_before:
        raise ValueError('Detector did not learn any parameter update')
    updates=[int(v['step']) for v in optimizer.state_dict()['state'].values() if 'step' in v]
    if not updates or min(updates)!=steps or max(updates)!=steps:
        raise ValueError('Exactly the declared real optimizer updates required')
    path=args.output/'last.pt'
    save(steps)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if any(not torch.equal(v.detach().cpu(),restored['model'][k]) for k,v in student.state_dict().items()):
        raise ValueError('Saved detector parameters do not reload exactly')
    del student,teacher,optimizer,images,predictions,native,reflected,objective
    torch.cuda.empty_cache()
    control=run_smoke(args.checkpoint,args.data/stems[0],args.output/'control',standalone_image_flow=True,
        pre_motion_patch=install_detector_spatial_tta,encode_patch=attention_guard)
    candidate=run_smoke(path,args.data/stems[0],args.output/'candidate',standalone_image_flow=True,
        pre_motion_patch=install_detector_spatial_tta,encode_patch=attention_guard)
    if min(control['predicted_nodes'],candidate['predicted_nodes'])<=0:
        raise ValueError('Nonempty detector outputs required')
    result=dict(status='passed_owned_detector_pu_functionality' if profile=='probe' else 'completed_owned_detector_fit_not_selection',steps=steps,identity=identity,history=history,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_hashes=input_hashes(),
        target_hashes=target_hashes,teacher_unchanged=True,linker_unchanged=True,bn_unchanged=True,detector_changed=True,
        strict_reload_identical=True,control=control,candidate=candidate,authorized_for_submission=False,
        probe_replay_passed=probe is not None,
        first10_gate=first10_gate,
        caveat='Source training and three training frames only; not independent detection or tracking improvement')
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],checkpoint_sha256=result['checkpoint_sha256'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--logit-targets',action='store_true')
    parser.add_argument('--fit-profile',choices=('probe','full'),default='probe')
    parser.add_argument('--objective',choices=('pu','sparse'),default='pu')
    main(parser.parse_args())
