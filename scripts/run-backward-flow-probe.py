"""Evidence-gated image-motion training; no detections, tracking or submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def loss_profile(name,steps):
    if name == 'sparse':
        return dict(image_weight=0.,smoothness_weight=0.)
    if name == 'image' and steps == 100:
        return dict(image_weight=.25,smoothness_weight=.01)
    raise ValueError('Image objective requires its bounded100step probe; no unverified extension')


def main(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path[:0] = [str(args.repo/'scripts'),str(args.repo/'src'),str(args.runtime)]
    import train_unet_transformer as official
    from backward_flow_model import BackwardFlowNet,observed_targets
    from backward_flow_ops import sparse_backward_loss,sample_backward_flow
    from seeded_frame_dataset import seed_dataset
    from independent_real_baseline import fresh_batches
    profile = loss_profile(getattr(args,'loss_profile','sparse'),args.steps)
    image_objective = profile['image_weight'] > 0
    if image_objective:
        from backward_flow_image_loss import image_alignment,physical_smoothness
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Two T4 GPUs required for the bounded probe')
    torch.set_num_threads(2)
    seed = 20260910
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    split_bytes = args.manifest.read_bytes()
    fold = json.loads(split_bytes)['folds'][0]
    diagnostic_stems = fold['train'][::5]
    fitting_stems = [s for s in fold['train'] if s not in diagnostic_stems]
    if (len(fitting_stems) != 96 or len(diagnostic_stems) != 24
        or set(fold['train'])&set(fold['selection']+fold['audit_order'])):
        raise ValueError('Frozen training-only scope required')
    probe = None
    if args.steps == 1000:
        probe = json.loads((args.runtime/'probe_result.json').read_text())
        if (probe['small_fit_gate_passed'] is not True or probe['steps'] != 100
            or probe['checkpoint_sha256'] != 'c1203ffb03ac98b4aa5b332ec26b4469ba5b9e784e6e095f03a41c533bba5884'
            or probe['identity']['split_sha256'] != hashlib.sha256(split_bytes).hexdigest()
            or probe['identity']['fitting_stems'] != fitting_stems):
            raise ValueError('Verified successful100step probe required')
    args.output.mkdir(parents=True,exist_ok=True)
    identity = dict(run_id='backward-flow-probe-v1' if args.steps == 100 else 'backward-flow-fit-v1',seed=seed,max_steps=args.steps,
        fitting_stems=fitting_stems,diagnostic_stems=diagnostic_stems,
        split_sha256=hashlib.sha256(split_bytes).hexdigest(),public_checkpoint_loaded=False,
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False,
        architecture='Owned residual3DUNet widths16,32,64,128; GroupNorm; pair channels; zero flow head',
        flow_convention='Current-grid ZYX physical microns; parent minus child',
        loss='Observed incoming-link coordinate-mean L1; unknown columns ignored',
        downsample=[1,4,4],batch_size=2,optimizer=dict(name='AdamW',lr=1e-4,weight_decay=.01),
        diagnostic_scope='First annotated two-frame window per24 training-only diagnostic movies; not tracking validation')
    if image_objective:
        identity.update(run_id='backward-flow-image-probe-v1',loss_profile=profile,
            loss='Observed incoming-link L1 +0.25 local3D image SSIM/boundary +0.01 physical smoothness')
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2))
    fitting,diagnostic = [],[]
    for stem in fold['train']:
        vm,windows = official.load_dataset_windows(args.data/stem,window_size=2,downsample=(1,4,4))
        if not windows:
            raise ValueError('Missing training windows in '+stem)
        if not np.allclose(vm.voxel_size,[1.625]*3,rtol=0,atol=1e-6):
            raise ValueError('Unexpected already-downsampled voxel size')
        if stem in diagnostic_stems:
            observable = next((w for w in windows if bool(w.targets[0].sum() > 0)),None)
            if observable is None:
                raise ValueError('No diagnostic window with incoming-link supervision')
            diagnostic.append((vm,[observable]))
        else:
            fitting.append((vm,windows))
    train = official.FrameWindowDataset(fitting,augmentations=official.DEFAULT_AUGMENTATIONS)
    diag = official.FrameWindowDataset(diagnostic,augmentations=[])
    training_displacements = []
    for meta,vm in train._data:
        points,target,mask = observed_targets(meta['coords'][None],meta['masks'][None],
            meta['targets'][None],torch.tensor([vm.voxel_size]))
        extent = points.new_tensor(vm.image_shape[1:])-1
        previous = points+target/points.new_tensor(vm.voxel_size)
        inside = ((points>=0)&(points<=extent)&(previous>=0)&(previous<=extent)).all(-1)
        training_displacements.append(target[mask&inside].numpy())
    all_displacements = np.concatenate(training_displacements)
    if not len(all_displacements):
        raise ValueError('No physical training displacements')
    median_um = np.median(all_displacements,axis=0)
    identity.update(training_median_um=median_um.tolist(),median_fit_links=len(all_displacements))
    seed_dataset(official,train,seed)
    seed_dataset(official,diag,seed+1)
    loader_rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(train,batch_size=2,shuffle=True,num_workers=0,generator=loader_rng)
    diag_loader = DataLoader(diag,batch_size=1,shuffle=False,num_workers=0)
    core = BackwardFlowNet().cuda()
    model = torch.nn.DataParallel(core)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=1024.)
    identity.update(parameters=sum(p.numel() for p in model.parameters()),fitting_windows=len(train))
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2))

    def targets(batch,device):
        points,target,mask = observed_targets(batch['coords'].to(device),batch['masks'].to(device),
            batch['targets'].to(device),batch['voxel_size'].to(device))
        extent = points.new_tensor(batch['imgs'].shape[2:])-1
        previous = points+target/batch['voxel_size'].to(device)[:,None]
        inside = ((points>=0)&(points<=extent)&(previous>=0)&(previous<=extent)).all(-1)
        excluded = int((mask&~inside).sum())
        return points,target,mask&inside,excluded

    def diagnose():
        model.eval()
        sums = dict(n=0,absolute=0.,endpoint=0.,zero_absolute=0.,zero_endpoint=0.,
                    median_absolute=0.,median_endpoint=0.,boundary_excluded=0)
        rows = []
        with torch.no_grad():
            for stem,batch in zip(diagnostic_stems,diag_loader):
                points,target,mask,excluded = targets(batch,'cuda')
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    flow = model(batch['imgs'].cuda().float())
                prediction,valid = sample_backward_flow(flow,points)
                if (mask&~valid).any():
                    raise ValueError('Diagnostic domain mask mismatch')
                delta = prediction[mask]-target[mask]
                median_delta = target[mask]-target.new_tensor(median_um)
                n = int(mask.sum())
                if n == 0:
                    raise ValueError('Diagnostic window has no observable incoming links: '+stem)
                row = dict(stem=stem,n=n,absolute=float(delta.abs().sum()),endpoint=float(delta.norm(dim=-1).sum()),
                    zero_absolute=float(target[mask].abs().sum()),zero_endpoint=float(target[mask].norm(dim=-1).sum()),
                    median_absolute=float(median_delta.abs().sum()),median_endpoint=float(median_delta.norm(dim=-1).sum()),
                    boundary_excluded=excluded)
                rows.append(row)
                for key in sums:
                    sums[key] += row[key]
        metrics = dict(mae_um=sums['absolute']/(3*sums['n']),endpoint_um=sums['endpoint']/sums['n'],
            zero_mae_um=sums['zero_absolute']/(3*sums['n']),zero_endpoint_um=sums['zero_endpoint']/sums['n'],
            median_mae_um=sums['median_absolute']/(3*sums['n']),median_endpoint_um=sums['median_endpoint']/sums['n'])
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError('Nonfinite motion diagnostic')
        return dict(**metrics,counts=sums,per_movie=rows)

    def save(step):
        path = args.output/'last.tmp.pt'
        torch.save(dict(model=core.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            identity=identity,step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            loader_rng=loader_rng.get_state(),augmentation_rng=train.augmentation_rng.bit_generator.state),path)
        os.replace(path,args.output/'last.pt')

    before = diagnose()
    if abs(before['mae_um']-before['zero_mae_um']) > 1e-6:
        raise ValueError('Zero-flow initialization diagnostic mismatch')
    save(0)
    stream = fresh_batches(loader)
    history = []
    started = time.monotonic()
    model.train()
    step100 = None
    for step in range(1,args.steps+1):
        batch = next(stream)
        points,target,mask,excluded = targets(batch,'cuda')
        if not mask.any():
            raise ValueError('No real incoming-link supervision in training batch')
        optimizer.zero_grad(set_to_none=True)
        images = batch['imgs'].cuda().float()
        with torch.amp.autocast('cuda',dtype=torch.float16):
            flow = model(images)
        sparse_loss = sparse_backward_loss(flow,points,target,mask)
        loss = sparse_loss
        image_terms = {}
        if image_objective:
            if not torch.allclose(batch['voxel_size'],batch['voxel_size'][0].expand_as(batch['voxel_size'])):
                raise ValueError('Image objective requires common physical grid per batch')
            voxel = batch['voxel_size'][0].tolist()
            alignment = image_alignment(images[:,0:1],images[:,1:2],flow,voxel)
            smoothness = physical_smoothness(flow,voxel)
            loss = loss + profile['image_weight']*alignment['loss'] + profile['smoothness_weight']*smoothness
            image_terms = dict(sparse_loss_um=float(sparse_loss.detach()),
                image_loss=float(alignment['loss'].detach()),ssim_loss=float(alignment['ssim_loss'].detach()),
                boundary_loss=float(alignment['boundary_loss'].detach()),smoothness=float(smoothness.detach()),
                texture_patches=alignment['texture_patches'],invalid_texture_patches=alignment['invalid_texture_patches'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite physical motion loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        row = dict(step=step,loss_um=float(loss.detach()),observed_links=int(mask.sum()),
            boundary_excluded=excluded,grad_norm=float(grad),elapsed_seconds=time.monotonic()-started)
        if image_objective:
            row['total_objective'] = row.pop('loss_um')
            row.update(image_terms)
        history.append(row)
        if step%10 == 0:
            print(json.dumps(row),flush=True)
        if step%25 == 0:
            save(step)
        if step == 100 and args.steps > 100:
            if train.sample_hashes != probe['input_hashes']:
                raise ValueError('Frozen probe inputs were not replayed')
            step100 = diagnose()
            if not (step100['mae_um'] < min(step100['zero_mae_um'],step100['median_mae_um'])
                and step100['endpoint_um'] < min(step100['zero_endpoint_um'],step100['median_endpoint_um'])):
                raise ValueError('Step100 motion-learning gate no longer passes')
            print(json.dumps(dict(step100_gate_passed=True,probe_inputs_replayed=True)),flush=True)
            model.train()
    after = diagnose()
    loaded = torch.load(args.output/'last.pt',map_location='cuda',weights_only=True)
    core.load_state_dict(loaded['model'],strict=True)
    reloaded = diagnose()
    if after != reloaded:
        raise ValueError('Strict checkpoint reload changed deterministic diagnostics')
    state_steps = [int(s['step']) for s in loaded['optimizer']['state'].values() if 'step' in s]
    if not state_steps or min(state_steps) != args.steps or max(state_steps) != args.steps:
        raise ValueError('Missing real optimizer updates')
    result = dict(status='completed_probe_not_tracking_candidate',identity=identity,steps=args.steps,
        checkpoint_sha256=hashlib.sha256((args.output/'last.pt').read_bytes()).hexdigest(),history=history,
        before=before,after=after,strict_reload_identical=True,input_hashes=train.sample_hashes,
        small_fit_gate_passed=(after['mae_um'] < min(after['zero_mae_um'],after['median_mae_um'])
            and after['endpoint_um'] < min(after['zero_endpoint_um'],after['median_endpoint_um'])),
        authorized_for_submission=False)
    if probe is not None:
        result.update(step100_diagnostic=step100,probe_inputs_replayed=train.sample_hashes[:200] == probe['input_hashes'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','checkpoint_sha256','small_fit_gate_passed')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,choices=(100,1000),default=100)
    parser.add_argument('--loss-profile',choices=('sparse','image'),default='sparse')
    main(parser.parse_args())
