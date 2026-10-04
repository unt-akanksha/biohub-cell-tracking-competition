"""Bounded association-only repair with a 100-step GPU functionality gate."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time


def training_scope(split, profile):
    fold = split['folds'][0]
    stems = fold['train'][:4] if profile == 'pilot' else fold['train']
    expected = 4 if profile == 'pilot' else 120
    if profile not in ('pilot','full') or len(stems) != expected or len(set(stems)) != expected:
        raise ValueError('Unexpected frozen training scope')
    if set(stems) & set(fold['selection']+fold['audit_order']):
        raise ValueError('Selection/audit overlap')
    if any(s.split('_')[0] != fold['training_embryo'] for s in stems):
        raise ValueError('Target embryo cannot enter training')
    return stems


def validate_joint_initialization(state,stems,motion_contract):
    identity = state['identity']
    if (state['step'] != 1000 or identity['max_steps'] != 1000
        or identity.get('training_profile') != 'full' or identity['training_stems'] != stems
        or identity.get('motion_residual') != motion_contract
        or identity['loss'] != 'sparse_parent_classification_with_null_v1'):
        raise ValueError('Completed full-source parent-only residual initialization required')


def validate_division_initialization(state,stems,motion_contract):
    identity = state['identity']
    if (state['step'] != 1000 or identity['max_steps'] != 1000
        or identity.get('training_profile') != 'full' or identity['training_stems'] != stems
        or identity.get('motion_residual') != motion_contract
        or identity.get('known_null') != dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False)
        or identity['loss'] != 'sparse_parent_with_annotated_missing_parent_null_v1'
        or identity.get('frozen_modules') != ['unet','detect_head']):
        raise ValueError('Completed known-null frozen-detector initialization required')


def main(args):
    division_specialist = getattr(args,'division_specialist',False)
    image_motion = getattr(args,'image_motion',False)
    if sum(bool(x) for x in (division_specialist,image_motion,getattr(args,'known_null',False))) > 1:
        raise ValueError('Specify one profile; specialist and image-motion profiles retain null supervision')
    known_null = getattr(args, 'known_null', False) or division_specialist or image_motion
    expected_initial = ('b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef' if division_specialist or image_motion
                        else 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470')
    if known_null and (args.joint or args.row_negatives or not args.motion_residual
                       or args.training_scope != 'full' or args.steps not in (100,1000)
                       or args.sha256 != expected_initial):
        raise ValueError('Known-null profile requires frozen full-source joint initialization')
    if args.steps == 6000 and (not args.joint or not args.checkpoint_bn_once):
        raise ValueError('Long profile requires joint optimization and the tested BN guard')
    if args.checkpoint_bn_once and not args.joint:
        raise ValueError('BN checkpoint guard is a joint-training-only profile')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError('Initialization checkpoint checksum mismatch')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path[:0] = [str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import train_unet_transformer as official
    from independent_real_baseline import fresh_batches, install_empty_attention_guard, patch_train_epoch
    from sparse_parent_loss import sparse_parent_loss
    from real_checkpoint_gpu_smoke import run_smoke
    from seeded_frame_dataset import seed_dataset
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Two CUDA devices required')
    torch.set_num_threads(2)
    random.seed(20260909); np.random.seed(20260909); torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    split = json.loads(args.manifest.read_text())
    stems = training_scope(split,args.training_scope)
    if args.joint or known_null:
        if args.training_scope != 'full' or not args.motion_residual or args.row_negatives:
            raise ValueError('Joint experiment requires full scope and the parent-only residual objective')
        from motion_residual import contract
        if division_specialist or image_motion:
            validate_division_initialization(state,stems,contract())
        else:
            validate_joint_initialization(state,stems,contract())
    elif state['step'] != 1000 or state['identity']['training_stems'] != split['folds'][0]['train'][:4]:
        raise ValueError('Frozen four-movie initialization required')
    if set(stems) & set(split['folds'][0]['selection']+split['folds'][0]['audit_order']):
        raise ValueError('Selection/audit overlap')
    identity = dict(run_id='independent-association-pilot-v1',training_stems=stems,
        training_profile=args.training_scope,
        initialization_sha256=args.sha256,loss='sparse_parent_classification_v1',
        frozen_modules=['unet','detect_head'],max_steps=args.steps,seed=20260909,
        optimizer=dict(name='AdamW',lr=1e-4,weight_decay=.01),
        augmentation_seed=20260909,augmentation_rng='independent PCG64',input_hashes_recorded=True,
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    if args.motion_residual:
        from motion_residual import contract,install_motion_residual,add_null_target,parent_probabilities
        identity.update(motion_residual=contract(),loss='sparse_parent_classification_with_null_v1',
            residual_head_initialization='zero')
    if args.row_negatives:
        if not args.motion_residual:
            raise ValueError('Row-negative profile requires the frozen motion/null contract')
        from sparse_parent_row_loss import sparse_parent_row_loss
        identity.update(loss='sparse_parent_and_row_hard_negative_v1',row_hard_negatives=4)
    if args.joint:
        identity.update(joint_training=True,frozen_modules=[],det_loss_weight=1.,det_neg_weight=.01,
            residual_head_initialization='preserved from hash-bound parent-only checkpoint',
            grad_scaler_initial_scale=1024.,optimizer_scope='all model parameters; fresh optimizer')
    if args.checkpoint_bn_once:
        identity.update(checkpoint_bn_once=True,
            continuation='Weight initialization only; fresh optimizer and recorded sampling seed')
    if known_null:
        identity.update(loss='sparse_parent_with_annotated_missing_parent_null_v1',
            known_null=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False),
            residual_head_initialization='preserved from hash-bound joint checkpoint')
    if division_specialist:
        identity.update(loss='division_balanced_with_annotated_missing_parent_null_v1',
            division_specialist=dict(version=1,division_window_mass=.5),
            residual_head_initialization='preserved from hash-bound known-null checkpoint')
    flow_core = None
    if image_motion:
        from image_motion_residual import contract as image_contract,validate_flow_initialization,flow_hash,install_image_motion_residual
        from backward_flow_model import BackwardFlowNet
        flow_path = getattr(args,'flow_checkpoint',None)
        if flow_path is None or hashlib.sha256(flow_path.read_bytes()).hexdigest() != image_contract()['flow_checkpoint_sha256']:
            raise ValueError('Exact frozen image-flow checkpoint required')
        flow_state = torch.load(flow_path,map_location='cpu',weights_only=True)
        validate_flow_initialization(flow_state,split)
        flow_core = BackwardFlowNet().cuda()
        flow_core.load_state_dict(flow_state['model'],strict=True)
        flow_core.requires_grad_(False).eval()
        del flow_state
        identity.update(image_motion=image_contract(),frozen_flow_sha256=flow_hash(flow_core),
            residual_head_initialization='preserved from b642; frozen image-flow spatial prior')
        if args.steps > 100:
            probe_path = args.runtime/'image_motion_probe.json'
            image_probe = json.loads(probe_path.read_text())
            if (image_probe['checkpoint_sha256'] != '8dc6e45657c063924cd84d90d8a520c9b7c736f398a2b2662b647645bd78dce4'
                or image_probe.get('flow_unchanged') is not True or image_probe.get('detector_unchanged') is not True
                or image_probe['identity']['frozen_flow_sha256'] != identity['frozen_flow_sha256']):
                raise ValueError('Verified matching image-motion probe required before extension')
            identity['image_motion_probe_sha256'] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2))
    model = official.UNetNodeTransformer(official.TemporalUNet3D(
        in_channels=1,out_channels=32,layers=[32,64,128]),unet_out_channels=32,pos_feat_dim=32).cuda()
    model.load_state_dict(state['model'],strict=True)
    del state
    if args.checkpoint_bn_once:
        from checkpoint_bn_guard import install_checkpoint_batchnorm_guard
        install_checkpoint_batchnorm_guard(model.unet)
    initial_bn_counts = {name: int(value) for name,value in model.state_dict().items()
                         if name.endswith('num_batches_tracked')}
    if not args.joint:
        for module in (model.unet,model.detect_head):
            module.requires_grad_(False)
    model.unet = torch.nn.DataParallel(model.unet)
    install_empty_attention_guard(model)
    if args.motion_residual:
        if not args.joint and not known_null:
            torch.nn.init.zeros_(model.transformer.pair_mlp[-1].weight)
            torch.nn.init.zeros_(model.transformer.pair_mlp[-1].bias)
        install_motion_residual(model)
    encode = model.encode
    def frozen_encode(images):
        model.unet.eval(); model.detect_head.eval()
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
            features,logits = encode(images)
        return features.float(),[v.float() for v in logits]
    model.encode = frozen_encode
    if image_motion:
        install_image_motion_residual(model,torch.nn.DataParallel(flow_core))
    if args.joint:
        def joint_encode(images):
            with torch.amp.autocast('cuda',dtype=torch.float16):
                features,logits = encode(images)
            return features.float(),[v.float() for v in logits]
        model.encode = joint_encode
    def detector_hash():
        digest = hashlib.sha256()
        for name,value in model.state_dict().items():
            if name.startswith(('unet.','detect_head.')):
                digest.update(name.encode()+b'\0')
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    frozen_hash = detector_hash()
    def module_hashes():
        return {name:hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes()
                    for p in module.parameters())).hexdigest()
                for name,module in [('unet',model.unet),('detect_head',model.detect_head),
                                    ('transformer',model.transformer)]}
    initial_module_hashes = module_hashes() if args.joint else None
    optimizer = torch.optim.AdamW(model.parameters() if args.joint else model.transformer.parameters(),
                                  lr=1e-4,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=1024.) if args.joint else None
    def finite_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise RuntimeError('Nonfinite association gradient')
        return gradient
    if not args.joint:
        for parameter in model.transformer.parameters():
            parameter.register_hook(finite_gradient)
    def save(step):
        payload = dict(model={k.replace('unet.module.','unet.',1):v for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),step=step,identity=identity,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            loader_rng=generator.get_state(),python_rng=random.getstate())
        payload.update(augmentation_rng=dataset.augmentation_rng.bit_generator.state,
                       sample_hashes=list(dataset.sample_hashes))
        if image_motion:
            payload['frozen_flow_model'] = flow_core.state_dict()
        if scaler is not None:
            payload['scaler'] = scaler.state_dict()
        torch.save(payload,args.output/'last.tmp.pt')
        os.replace(args.output/'last.tmp.pt',args.output/'last.pt')
    videos = [official.load_dataset_windows(args.data/s,window_size=2,downsample=(1,4,4)) for s in stems]
    windows = [w for _,items in videos for w in items]
    maximum = max(max(w.node_counts) for w in windows)
    dimension = windows[0].pos_feats[0].shape[1]
    metadata_bytes = sum(w.n_frames*maximum*(dimension*4+12+1)+w.n_frames*8+
                         (w.n_frames-1)*maximum*maximum*4 for w in windows)
    if metadata_bytes > 4*1024**3:
        raise RuntimeError('Padded training metadata exceeds 4 GiB guard')
    metadata_profile = dict(movies=len(videos),windows=len(windows),max_gt_nodes=int(maximum),
                           estimated_padded_tensor_bytes=int(metadata_bytes),image_loading='lazy Zarr')
    (args.output/'dataset_profile.json').write_text(json.dumps(metadata_profile,indent=2))
    print(json.dumps(dict(event='dataset_profile',**metadata_profile)),flush=True)
    dataset = official.FrameWindowDataset(videos,augmentations=official.DEFAULT_AUGMENTATIONS)
    seed_dataset(official,dataset,20260909)
    generator = torch.Generator().manual_seed(20260909)
    if division_specialist:
        from division_window_sampling import division_flags, balanced_weights
        from torch.utils.data import WeightedRandomSampler
        flags = division_flags(meta['targets'].numpy() for meta,_ in dataset._data)
        weights = balanced_weights(flags)
        sampler = WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(dataset),
                                        replacement=True,generator=generator)
        loader = DataLoader(dataset,batch_size=2,sampler=sampler,num_workers=0,generator=generator)
        identity['division_sampling'] = dict(division_windows=int(flags.sum()),ordinary_windows=int((~flags).sum()),
            division_mass=float(weights[flags].sum()),replacement=True,
            weights_sha256=hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest())
        (args.output/'identity.json').write_text(json.dumps(identity,indent=2))
        print(json.dumps(dict(event='division_sampling',**identity['division_sampling'])),flush=True)
    else:
        loader = DataLoader(dataset,batch_size=2,shuffle=True,num_workers=0,generator=generator)
    scope = dict(vars(official),fresh_batches=fresh_batches)
    epoch_source = inspect.getsource(official.train_epoch)
    if epoch_source.count('_cycle(loader)') != 1:
        raise ValueError('Organizer train-epoch source drift')
    if args.joint:
        scope['scaler'] = scaler
        epoch_source = patch_train_epoch(epoch_source)
    else:
        epoch_source = epoch_source.replace('_cycle(loader)','fresh_batches(loader)')
    if known_null:
        from annotated_missing_parent import patch_epoch, batch_missing_parent_masks
        from sparse_parent_missing_null import missing_null_targets
        epoch_source = patch_epoch(epoch_source)
        scope['batch_missing_parent_masks'] = batch_missing_parent_masks
    if division_specialist:
        from division_missing_null_loss import division_missing_null_loss
    exec(compile(epoch_source,'<association-epoch>','exec'),scope)
    telemetry = dict(correct=0,positive_links=0,confident_correct=0,max_nodes=0,wrong_child_claims=0,
                     known_null_columns=0,confident_null_columns=0,division_columns=0,correct_division_columns=0)
    def batch_loss(logits,target,mask_src,mask_tgt,*,known_null=None):
        losses = []
        for b in range(logits.shape[0]):
            ns,nt = int(mask_src[b].sum()),int(mask_tgt[b].sum())
            raw,truth = logits[b,:ns,:nt],target[b,:ns,:nt]
            if known_null is not None:
                augmented, augmented_truth = missing_null_targets(raw,truth,known_null[b],
                    identity['motion_residual']['null_logit'])
                losses.append(division_missing_null_loss(raw,truth,known_null[b]) if division_specialist
                              else sparse_parent_loss(augmented,augmented_truth))
                with torch.no_grad():
                    null_mask = augmented_truth[-1].bool()
                    telemetry['known_null_columns'] += int(null_mask.sum())
                    telemetry['confident_null_columns'] += int((torch.softmax(augmented,dim=0)[-1,null_mask] > .5).sum())
            elif args.row_negatives:
                losses.append(sparse_parent_row_loss(raw,truth,hard_negatives=4))
            else:
                losses.append(sparse_parent_loss(*add_null_target(raw,truth)) if args.motion_residual
                              else sparse_parent_loss(raw,truth))
            active = truth.sum(0) == 1
            if ns and active.any():
                with torch.no_grad():
                    probs = parent_probabilities(raw[:,active]) if args.motion_residual else torch.softmax(raw[:,active],dim=0)
                    chosen = probs.argmax(0)
                    expected = truth[:,active].argmax(0)
                    telemetry['positive_links'] += int(active.sum())
                    telemetry['correct'] += int((chosen == expected).sum())
                    telemetry['confident_correct'] += int(((chosen == expected)&(probs.max(0).values > .5)).sum())
                    division_columns = ((truth > 0)&(truth.sum(1) == 2).unsqueeze(1)).any(0)[active]
                    telemetry['division_columns'] += int(division_columns.sum())
                    telemetry['correct_division_columns'] += int(((chosen == expected)&division_columns).sum())
                    if args.motion_residual:
                        wrong = (truth.sum(1) > 0).unsqueeze(1)&(truth == 0)
                        telemetry['wrong_child_claims'] += int(((parent_probabilities(raw) > .5)&wrong).sum())
        return torch.stack(losses).mean()
    original_match = official.detect_and_match
    def guarded_match(*values,**kwargs):
        result = original_match(*values,**kwargs)
        maximum = int(result[2].sum(1).max())
        telemetry['max_nodes'] = max(telemetry['max_nodes'],maximum)
        if maximum > 2048:
            raise RuntimeError('Attention memory guard exceeded; no truncation')
        return result
    scope['compute_batch_loss'] = batch_loss
    scope['detect_and_match'] = guarded_match
    save(0)
    before = run_smoke(args.output/'last.pt',args.data/stems[0],args.output/'before')
    started = time.monotonic()
    history = []
    step100_smoke = None
    for step in range(10,args.steps+1,10):
        if step > 100 and (step100_smoke is None or step100_smoke['status'] != 'passed'):
            raise RuntimeError('Missing step-100 GPU functionality gate')
        if time.monotonic()-started > (6300 if args.steps == 6000 else 2700 if args.joint else 1200):
            raise RuntimeError('Small association smoke time budget exceeded')
        telemetry.update(correct=0,positive_links=0,confident_correct=0,max_nodes=0,wrong_child_claims=0,
                         known_null_columns=0,confident_null_columns=0,division_columns=0,correct_division_columns=0)
        edge,det = scope['train_epoch'](model,loader,optimizer,torch.device('cuda'),
            det_loss_weight=1. if args.joint else 0.,det_neg_weight=.01,max_iters=10,pool_kernel_um=5.)
        if not np.isfinite(edge) or not np.isfinite(det):
            raise RuntimeError('Nonfinite parent loss')
        save(step)
        row = dict(step=step,parent_loss=edge,detector_loss=det,**telemetry,
            elapsed_seconds=time.monotonic()-started)
        if args.joint:
            updates = [int(v['step']) for v in optimizer.state.values() if 'step' in v]
            row.update(optimizer_steps_min=min(updates,default=0),optimizer_steps_max=max(updates,default=0),
                grad_scaler_scale=scaler.get_scale(),
                gpu_peak_allocated=[torch.cuda.max_memory_allocated(i) for i in range(2)])
        if args.checkpoint_bn_once:
            deltas = {name.replace('unet.module.','unet.',1): int(value) -
                      initial_bn_counts[name.replace('unet.module.','unet.',1)]
                      for name,value in model.state_dict().items() if name.endswith('num_batches_tracked')}
            if not deltas or any(value != step for value in deltas.values()):
                raise ValueError('BatchNorm counters must advance exactly once per training batch')
            row['batchnorm_update_deltas'] = deltas
        history.append(row)
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        if step == 100 and args.steps > 100:
            import shutil
            if known_null and sum(r['known_null_columns'] for r in history) == 0:
                raise ValueError('No real annotated missing-parent supervision observed; do not extend')
            if division_specialist and sum(r['division_columns'] for r in history) == 0:
                raise ValueError('No matched two-daughter supervision observed; do not extend')
            if not args.joint and detector_hash() != frozen_hash:
                raise ValueError('Frozen detector changed before extension')
            if image_motion and flow_hash(flow_core) != identity['frozen_flow_sha256']:
                raise ValueError('Frozen image-flow model changed before extension')
            if image_motion and dataset.sample_hashes != image_probe['sample_hashes']:
                raise ValueError('Image-motion probe inputs did not replay before extension')
            if args.joint:
                current = module_hashes()
                if row['optimizer_steps_min'] < 90 or any(current[k] == v for k,v in initial_module_hashes.items()):
                    raise ValueError('Joint optimization did not update every module before extension')
            shutil.copyfile(args.output/'last.pt',args.output/'step100.pt')
            step100_smoke = run_smoke(args.output/'step100.pt',args.data/stems[0],args.output/'step100')
            (args.output/'step100_smoke.json').write_text(json.dumps(step100_smoke,indent=2))
    if not args.joint and detector_hash() != frozen_hash:
        raise ValueError('Frozen detector changed')
    if args.joint:
        final_module_hashes = module_hashes()
        if row['optimizer_steps_min'] < .9*args.steps or any(final_module_hashes[k] == v for k,v in initial_module_hashes.items()):
            raise ValueError('Joint optimization incomplete or a module stayed frozen')
    after = run_smoke(args.output/'last.pt',args.data/stems[0],args.output/'after')
    result = dict(status='completed',identity=identity,history=history,
        before=before,after=after,step100_smoke=step100_smoke,
        initial_detector_sha256=frozen_hash,detector_unchanged=detector_hash() == frozen_hash,
        checkpoint_sha256=hashlib.sha256((args.output/'last.pt').read_bytes()).hexdigest(),
        sample_hashes=dataset.sample_hashes,
        authorized_for_submission=False)
    if image_motion:
        result['flow_unchanged'] = flow_hash(flow_core) == identity['frozen_flow_sha256']
        if not result['flow_unchanged']:
            raise ValueError('Frozen image-flow model changed during linker fit')
    if args.joint:
        result.update(initial_module_hashes=initial_module_hashes,final_module_hashes=final_module_hashes)
    else:
        result['frozen_detector_sha256'] = frozen_hash
    if known_null:
        result['known_null_supervised_total'] = sum(r['known_null_columns'] for r in history)
    if division_specialist:
        result['division_supervised_total'] = sum(r['division_columns'] for r in history)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--steps',type=int,choices=(100,1000,6000),default=100)
    parser.add_argument('--training-scope',choices=('pilot','full'),default='pilot')
    parser.add_argument('--motion-residual',action='store_true')
    parser.add_argument('--row-negatives',action='store_true')
    parser.add_argument('--joint',action='store_true')
    parser.add_argument('--checkpoint-bn-once',action='store_true')
    parser.add_argument('--known-null',action='store_true')
    parser.add_argument('--division-specialist',action='store_true')
    parser.add_argument('--image-motion',action='store_true')
    parser.add_argument('--flow-checkpoint',type=Path)
    main(parser.parse_args())
