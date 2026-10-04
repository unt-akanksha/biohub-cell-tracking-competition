"""Bounded source-embryo optimization pilot; never loads public checkpoints."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def main(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path[:0] = [str(args.repo / 'scripts'), str(args.repo / 'src'), str(args.runtime)]
    import train_unet_transformer as official
    from independent_real_baseline import fresh_batches, patch_train_epoch, install_empty_attention_guard
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    fold = manifest['folds'][0]
    stems = fold['train'][:4]
    assert set(stems).isdisjoint(fold['selection'] + fold['audit_order'])
    assert all(s.startswith(fold['training_embryo'] + '_') for s in stems)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Pilot requires two T4 GPUs')
    random.seed(20260909); np.random.seed(20260909); torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    torch.set_num_threads(2)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(run_id='independent-real-pilot-v1', training_stems=stems,
        held_out_embryo=fold['held_out_embryo'], target_audit_opened=False,
        public_checkpoint_loaded=False, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        authorized_for_submission=False, seed=20260909,
        architecture=dict(unet_layers=[32,64,128], unet_out_channels=32, window_size=2,
                          downsample=[1,4,4], pool_kernel_um=5.),
        optimizer=dict(name='AdamW', lr=1e-4, weight_decay=.01), batch_size=2,
        det_loss_weight=1., det_neg_weight=.01, max_steps=args.steps,
        gpu_smoke_required=args.steps > 100,
        detector_warmup_steps=50, detector_initial_probability=.01,
        empty_attention_guard=True,
        initialization_revision='Zero detection-head weights and fixed sparse-positive bias; fresh backbone/linker')
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2))
    videos = []
    for stem in stems:
        videos.append(official.load_dataset_windows(args.data / stem,
            window_size=2, downsample=(1,4,4)))
    if any(not windows for _, windows in videos):
        raise RuntimeError('A frozen training movie has no annotated frame windows')
    dataset = official.FrameWindowDataset(videos, augmentations=official.DEFAULT_AUGMENTATIONS)
    generator = torch.Generator().manual_seed(20260909)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, generator=generator)
    model = official.UNetNodeTransformer(
        official.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32,64,128]),
        unet_out_channels=32, pos_feat_dim=4*official._POS_EMBED_DIM).cuda()
    # A fixed sparse-positive prior, unrelated to organizer estimated counts.
    torch.nn.init.zeros_(model.detect_head.weight)
    torch.nn.init.constant_(model.detect_head.bias, math.log(.01/.99))
    model.unet = torch.nn.DataParallel(model.unet)
    install_empty_attention_guard(model)
    original_encode = model.encode
    def amp_encode(imgs):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            features, logits = original_encode(imgs)
        return features.float(), [x.float() for x in logits]
    model.encode = amp_encode
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    scope = dict(vars(official), fresh_batches=fresh_batches, scaler=scaler)
    source = patch_train_epoch(inspect.getsource(official.train_epoch))
    exec(compile(source, '<pilot-amp-epoch>', 'exec'), scope)
    epoch = scope['train_epoch']
    original_match = official.detect_and_match
    counts = []
    coverage = {'matched':0, 'annotated':0, 'supervised_pairs':0}
    def observe_match(*pos, **kw):
        result = original_match(*pos, **kw)
        count = int(result[2].sum(dim=1).max().item())
        counts.append(count)
        coverage['matched'] += sum(int((matches >= 0).sum().item()) for matches in result[3])
        coverage['annotated'] += int(pos[2].sum().item())
        # Abort, never truncate detections or alter the objective to fit VRAM.
        if count > 2048:
            raise RuntimeError('Untrained detector exceeds pilot attention memory guard')
        return result
    scope['detect_and_match'] = observe_match
    original_batch_loss = official.compute_batch_loss
    def observe_batch_loss(logits, target, mask_t, mask_t1):
        coverage['supervised_pairs'] += int((target.flatten(1).sum(dim=1) > 0).sum().item())
        return original_batch_loss(logits, target, mask_t, mask_t1)
    scope['compute_batch_loss'] = observe_batch_loss
    history = []
    warmup_steps = 0
    def save(step):
        state = dict(model={k.replace('unet.module.', 'unet.', 1): v for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step,
            warmup_steps=warmup_steps,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            loader_rng=generator.get_state(), python_rng=random.getstate(),
            numpy_rng=dict(kind=np.random.get_state()[0], keys=np.random.get_state()[1].tolist(),
                           position=int(np.random.get_state()[2]), has_gauss=int(np.random.get_state()[3]),
                           cached_gaussian=float(np.random.get_state()[4])), identity=identity)
        temp = args.output / 'last.tmp.pt'
        torch.save(state, temp)
        os.replace(temp, args.output / 'last.pt')
    save(0)
    warmup_history = []
    batches = fresh_batches(loader)
    model.train()
    for warmup_steps in range(1, 51):
        if time.monotonic() - started >= 2800:
            raise RuntimeError('Detector warm-up exhausted pilot budget')
        batch = next(batches)
        images = batch['imgs'].cuda()
        coords, masks = batch['coords'].cuda(), batch['masks'].cuda()
        _, logits = model.encode(images)
        loss = sum(official.compute_detection_loss(logits[i], coords[:, i], masks[:, i],
                   neg_weight=.01) for i in range(images.shape[1])) / images.shape[1]
        if not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite detector warm-up loss')
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer); scaler.update()
        if warmup_steps % 10 == 0:
            save(0)
            warmup_history.append(dict(step=warmup_steps, loss=float(loss.detach()),
                                       elapsed_seconds=time.monotonic()-started))
            (args.output/'warmup_history.json').write_text(json.dumps(warmup_history, indent=2))
            print(json.dumps(dict(event='detector_warmup', **warmup_history[-1])), flush=True)
    # Joint optimizer accounting is independent of detector-only updates.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    del batches, batch, images, coords, masks, logits, loss, _
    save(0)
    gpu_smoke = None
    for step in range(10, args.steps+1, 10):
        if step > 100 and (gpu_smoke is None or gpu_smoke['status'] != 'passed'):
            raise RuntimeError('Cannot extend training without passed GPU reload/inference smoke')
        if time.monotonic() - started >= 3000:
            break
        counts.clear()
        coverage.update(matched=0, annotated=0, supervised_pairs=0)
        edge, detection = epoch(model, loader, optimizer, torch.device('cuda'),
            det_loss_weight=1., det_neg_weight=.01, max_iters=10, pool_kernel_um=5.)
        if not math.isfinite(edge) or not math.isfinite(detection):
            raise RuntimeError('Nonfinite training loss')
        save(step)
        row = dict(step=step, edge_loss=edge, detection_loss=detection,
            matched_training_nodes=coverage['matched'], annotated_training_nodes=coverage['annotated'],
            training_node_recall=coverage['matched']/max(1,coverage['annotated']),
            supervised_training_pairs=coverage['supervised_pairs'],
            grad_scaler_scale=scaler.get_scale(),
            optimizer_steps_max=max((int(v['step']) for v in optimizer.state.values() if 'step' in v), default=0),
            max_detected_nodes=max(counts, default=0), elapsed_seconds=time.monotonic()-started,
            gpu_peak_allocated=[torch.cuda.max_memory_allocated(i) for i in range(2)])
        history.append(row)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        if step == 100 and args.steps > 100:
            import shutil
            from real_checkpoint_gpu_smoke import run_smoke
            shutil.copyfile(args.output/'last.pt', args.output/'smoke_checkpoint.pt')
            gpu_smoke = run_smoke(args.output/'smoke_checkpoint.pt', args.data/stems[0], args.output)
            (args.output/'gpu_smoke.json').write_text(json.dumps(gpu_smoke, indent=2))
            print(json.dumps(dict(event='gpu_smoke', **gpu_smoke)), flush=True)
    result = dict(identity, status='completed_pilot', history=history,
        warmup_history=warmup_history,
        gpu_smoke=gpu_smoke,
        checkpoint_sha256=hashlib.sha256((args.output/'last.pt').read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started, trained_steps=history[-1]['step'] if history else 0,
        selection_opened=False, production_promotion_authorized=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'runtime', 'manifest', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=(100,1000), default=100)
    main(parser.parse_args())
