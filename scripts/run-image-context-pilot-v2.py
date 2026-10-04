"""Antelume-only 74.7M image-context smoke and embryo-excluded training pilot."""
import argparse
import copy
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, row):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def gpu_available(allow_self=False):
    value = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        check=True, capture_output=True, text=True, timeout=15).stdout
    pids = {int(x.strip()) for x in value.splitlines() if x.strip()}
    if pids - ({os.getpid()} if allow_self else set()):
        raise RuntimeError('Other GPU work exists; leave other projects untouched')


def execute(args, result):
    contract = json.loads((args.bundle / 'BUNDLE.json').read_text())
    if sha(args.bundle / 'BUNDLE.json') != args.bundle_sha256:
        raise ValueError('Frozen bundle changed')
    for name, digest in contract['files'].items():
        path = (args.bundle / name).resolve()
        if args.bundle.resolve() not in path.parents or sha(path) != digest:
            raise ValueError('Runtime input changed')
    if args.mode == 'pilot':
        smoke = json.loads(args.smoke_proof.read_text())
        if not (smoke['status'] == 'functionality_passed' and smoke['bundle_sha256'] == args.bundle_sha256
                and smoke['estimated_pilot_seconds_with_margin'] < 2400):
            raise ValueError('Same-bundle smoke and throughput gate required')
    gpu_available()
    if shutil.disk_usage(args.output).free < 3.7 * 1024**3:
        raise RuntimeError('Insufficient disk headroom for resumable checkpoints')
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path.insert(0, str(args.bundle))
    import numpy as np
    import torch
    from research.temporal_contrastive.graph_context_division_model import GraphContextDivisionModel, load_backbone_checkpoint
    from research.temporal_contrastive.train_graph_context_division_sweep import balanced_rows, augment_batch
    from research.temporal_contrastive.graph_context_training_support import focal_loss, update_ema
    from research.image_context_quality import metrics, utility
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One freed Antelume GPU required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data_manifest = json.loads((args.bundle / 'data/MANIFEST.json').read_text())
    device = torch.device('cuda:0')

    def load_data(embryo, role):
        name = f'{embryo}-{role}.npz'
        record = data_manifest['files'][name]
        path = args.bundle / 'data' / name
        if sha(path) != record['sha256']:
            raise ValueError('Data packet changed')
        inventory = json.loads((path.parent / record['inventory_path']).read_text())
        if any(x['embryo'] != embryo or x['role'] != role for x in inventory):
            raise ValueError('Embryo/role exclusion failed')
        with np.load(path, allow_pickle=False) as a:
            data = [torch.from_numpy(a[k].copy()) for k in ('patches', 'geometry', 'context', 'mask', 'targets', 'weights', 'eligible')]
        data[3] = data[3].bool(); data[6] = data[6].bool()
        return data

    @torch.inference_mode()
    def predict(model, data):
        model.eval(); output = []
        for start in range(0, len(data[0]), 20):
            with torch.autocast('cuda', dtype=torch.float16):
                value = model(*(x[start:start+20].to(device) for x in data[:4]))
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite predictions')
            output.append(value.float().cpu())
        return torch.cat(output)

    outcomes = {}
    stage_seconds = []
    for index, source in enumerate(('44b6', '6bba')[:1 if args.mode == 'smoke' else 2]):
        target = '6bba' if source == '44b6' else '44b6'
        train, selection = load_data(source, 'optimization'), load_data(source, 'selection')
        seed = 20260913 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed + 91337)
        model = GraphContextDivisionModel()
        if sum(p.numel() for p in model.parameters()) != 74732308:
            raise ValueError('Model capacity changed')
        warm = args.bundle / f'warm_start_target_{target}.pt'
        load_backbone_checkpoint(model, torch.load(warm, map_location='cpu', weights_only=True))
        model = model.to(device)
        ema = copy.deepcopy(model).requires_grad_(False).eval()
        head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
        optimizer = torch.optim.AdamW([dict(params=model.backbone.parameters(), lr=9e-6),
                                       dict(params=head, lr=6e-5)], weight_decay=2e-4)
        scaler = torch.amp.GradScaler('cuda', init_scale=4096.)
        history, best = [], None
        steps = 20 if args.mode == 'smoke' else 2000
        member = args.output / f'source-{source}'
        member.mkdir()
        start = time.monotonic()
        step_times = []
        for step in range(1, steps+1):
            tick = time.monotonic()
            model.train()
            rows = balanced_rows(train[4], train[6], 10, generator)
            patches, geometry, context = augment_batch(train[0][rows], train[1][rows], train[2][rows], generator)
            batch = [x.to(device) for x in (patches, geometry, context, train[3][rows])]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(*batch)
                loss = focal_loss(logits, train[4][rows].to(device), train[5][rows].to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradients')
            scaler.step(optimizer); scaler.update(); update_ema(model, ema, .995)
            lr = 3e-7 + .5 * (6e-5 - 3e-7) * (1 + math.cos(math.pi * step / 2000))
            optimizer.param_groups[0]['lr'] = lr * .15; optimizer.param_groups[1]['lr'] = lr
            torch.cuda.synchronize()
            step_times.append(time.monotonic()-tick)
            if step == 1 or step % 100 == 0 or step == steps:
                print(json.dumps(dict(event='training', source=source, step=step, loss=float(loss),
                                     elapsed_seconds=time.monotonic()-start)), flush=True)
            if step % 250 == 0 or step == steps:
                evaluation_start = time.monotonic()
                scores = predict(ema, selection)
                selected = selection[6]
                row = metrics(selection[4][selected].numpy(), scores[selected].numpy())
                if best is None or utility(row) > utility(best['metrics']):
                    best = dict(step=step, metrics=row)
                    torch.save(ema.state_dict(), member / 'best.partial')
                    (member / 'best.partial').replace(member / 'best.pt')
                history.append(dict(step=step, metrics=row))
                write_json(member / 'selection-history.json', dict(rows=history))
                checkpoint = dict(step=step, source=source, target=target,
                    model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), generator=generator.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    bundle_sha256=args.bundle_sha256, best=best)
                # One sequential owner, one rolling resume file; never touch another project.
                torch.save(checkpoint, args.output / 'latest.partial')
                (args.output / 'latest.partial').replace(args.output / 'latest.pt')
                del checkpoint
                stage_seconds.append(time.monotonic()-evaluation_start)
                print(json.dumps(dict(event='source_selection', source=source, step=step,
                                     **row)), flush=True)
                if args.mode == 'smoke':
                    checkpoint = torch.load(args.output / 'latest.pt', map_location='cpu', weights_only=True)
                    ema.load_state_dict(checkpoint['ema'], strict=True)
                    model.load_state_dict(checkpoint['model'], strict=True)
                    optimizer.load_state_dict(checkpoint['optimizer']); scaler.load_state_dict(checkpoint['scaler'])
                    generator.set_state(checkpoint['generator'])
                    replay = predict(ema, selection)
                    if not torch.equal(scores, replay):
                        raise ValueError('Checkpoint inference replay mismatch')
                    result['checkpoint_roundtrip_sha256'] = sha(args.output / 'latest.pt')
                    del checkpoint
                    torch.cuda.synchronize()
                    result['checkpoint_logits_exact'] = True
                gpu_available(allow_self=True)
        outcome = dict(source_embryo=source, held_out_embryo=target, steps=steps, best=best,
            checkpoint_sha256=sha(member / 'best.pt'), warm_start_sha256=sha(warm),
            elapsed_seconds=time.monotonic()-start, source_gate_passed=best['metrics']['source_gate_passed'])
        write_json(member / 'terminal.json', outcome)
        outcomes[source] = outcome
        if args.mode == 'smoke':
            # Include full warm starts, both-fold selection passes, and checkpoint I/O margin.
            estimate = float(np.quantile(step_times[5:], .9)) * 4000 * 1.5 + max(stage_seconds) * 16 * 1.5 + 120
            result.update(status='functionality_passed', optimizer_steps=20,
                parameter_count=74732308, steady_step_p90_seconds=float(np.quantile(step_times[5:], .9)),
                estimated_pilot_seconds_with_margin=estimate, pilot_runtime_gate_passed=estimate < 2400)
        del model, ema, optimizer, train, selection, scores, head, best, batch, logits, loss
        gc.collect();torch.cuda.empty_cache()
    result['source_models'] = outcomes
    if args.mode == 'smoke':
        # Exactly two generated verification files, not user artifacts or any recursive cleanup.
        for path in (args.output / 'latest.pt', args.output / 'source-44b6/best.pt'):
            if args.output.resolve() not in path.resolve().parents:
                raise ValueError('Unsafe temporary checkpoint cleanup')
            path.unlink()
        result['verified_temporary_checkpoint_files_removed'] = True
    elif not all(x['source_gate_passed'] for x in outcomes.values()):
        result.update(status='rejected_source_selection', held_out_embryo_scores_opened=False)
    else:
        # Freeze both source models and thresholds before opening target-embryo evaluation.
        write_json(args.output / 'frozen-source-policy.json', outcomes)
        target_rows = {}
        for source, outcome in outcomes.items():
            target = outcome['held_out_embryo']
            parts = [load_data(target, role) for role in ('optimization', 'selection')]
            data = [torch.cat([p[k] for p in parts]) for k in range(7)]
            model = GraphContextDivisionModel().to(device)
            model.load_state_dict(torch.load(args.output / f'source-{source}/best.pt', map_location='cpu', weights_only=True))
            scores = predict(model, data)
            mask = data[6]
            row = metrics(data[4][mask].numpy(), scores[mask].numpy(),
                          threshold=outcome['best']['metrics']['source_threshold'])
            target_rows[target] = row
            np.savez_compressed(args.output / f'held-out-{target}.npz', scores=scores.numpy(), labels=data[4].numpy(), eligible=mask.numpy())
            del model, data, parts, scores;gc.collect();torch.cuda.empty_cache()
        counts = [x['fixed_threshold'] for x in target_rows.values()]
        passed = (all(x['average_precision'] >= .55 and x['fixed_threshold']['tp'] >= 1 for x in target_rows.values())
                  and sum(x['tp'] for x in counts) >= 3 and sum(x['fp'] for x in counts) <= 1)
        result.update(status='passed_patch_pilot' if passed else 'rejected_embryo_transfer',
                      held_out=target_rows, held_out_embryo_scores_opened=True, patch_pilot_gate_passed=passed)
    result['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bundle-sha256', required=True)
    p.add_argument('--mode', choices=('smoke', 'pilot'), required=True)
    p.add_argument('--smoke-proof', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(run_id='image-division-context-v2', mode=args.mode, status='running',
        bundle_sha256=args.bundle_sha256, authorized_for_submission=False, kaggle_gpu_hours=0,
        competition_test_data_read=False, public_predictions_used=False, held_out_embryo_scores_opened=False)
    def timeout():
        write_json(args.output / 'timeout.json', dict(status='walltime_limit', resumable='latest.pt'))
        os._exit(124)
    timer = threading.Timer(600 if args.mode == 'smoke' else 2400, timeout)
    timer.daemon = True;timer.start()
    try:
        execute(args, result)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        timer.cancel(); result['elapsed_seconds'] = time.monotonic()-start
        write_json(args.output / 'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
