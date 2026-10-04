"""Bounded Antelume smoke -> sequential two-architecture/two-embryo training.

No global installs, submissions, foreign-process termination, or instance stop.
"""
from __future__ import annotations
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def gpu_pids():
    text = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=15).stdout
    return {int(line) for line in text.splitlines() if line.strip()}


def main(args, result):
    if sha(args.bundle/'BUNDLE.json') != args.bundle_sha256:
        raise ValueError('Bundle contract mismatch')
    contract = json.loads((args.bundle/'BUNDLE.json').read_text())
    for name, digest in contract['files'].items():
        if sha(args.bundle/name) != digest:
            raise ValueError('Runtime source changed')
    if args.mode == 'train':
        proof = json.loads(args.smoke_proof.read_text())
        if not (proof['status'] == 'functionality_passed' and proof['bundle_sha256'] == args.bundle_sha256
                and proof['estimated_four_member_seconds'] <= contract['total_watchdog_seconds']-480):
            raise ValueError('Matching passing smoke and throughput required')
    if gpu_pids():
        raise RuntimeError('GPU not free; leave other work intact')
    if shutil.disk_usage(args.output).free < 850*1024**2:
        raise RuntimeError('Need 850MiB checkpoint disk headroom')
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.path.insert(0, str(args.bundle))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from visual_correspondence_models_v1 import VisualCorrespondence, prepare_patches, mask_scores
    from visual_correspondence_data_v1 import validate_roles
    validate_roles(contract['records'])
    torch.set_num_threads(2)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Antelume GPU')
    torch.cuda.set_per_process_memory_fraction(.90)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    result['environment'] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), python=sys.version,
                                  gpu_memory_fraction=.90, cpu_threads=2)
    stop_requested = threading.Event()
    def monitor():
        while not stop_requested.wait(30):
            try:
                if gpu_pids() - {os.getpid()}:
                    result['foreign_gpu_detected'] = True; stop_requested.set()
            except Exception:
                result['gpu_check_failed'] = True; stop_requested.set()
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    data_cache = {}
    def data(embryo, role):
        key = embryo, role
        if key in data_cache:
            return data_cache[key]
        parts = {k: [] for k in ('patches', 'ids', 'coords', 'masks', 'targets')}
        movie_ids, movie_names, offset = [], [], 0
        for r in contract['records']:
            if (r['embryo'], r['role']) != key:
                continue
            path = args.bundle/r['path']
            if sha(path) != r['sha256']:
                raise ValueError('Training data hash mismatch')
            with np.load(path, allow_pickle=False) as packet:
                arrays = {k: packet[k].copy() for k in parts}
            if r['stem'] not in movie_names:
                movie_names.append(r['stem'])
            ids = arrays['ids']; ids[ids >= 0] += offset
            offset += len(arrays['patches'])
            for k, v in arrays.items():
                parts[k].append(torch.from_numpy(v))
            movie_ids.extend([movie_names.index(r['stem'])]*len(ids))
        out = {k: torch.cat(v).to(device) for k, v in parts.items()}
        out.update(movie_ids=torch.tensor(movie_ids, device=device), movie_names=movie_names)
        out['sampling_weights'] = torch.bincount(out['movie_ids'])[out['movie_ids']].float().reciprocal()
        data_cache[key] = out
        return out
    def batch(packet, rows, generator=None, augment=False, drop=False):
        ids = packet['ids'][rows].long(); valid = ids >= 0
        coords = packet['coords'][rows].clone(); mask = packet['masks'][rows].clone()
        targets = packet['targets'][rows].clone().long()
        if drop:
            chosen = targets < 16
            if augment:
                chosen &= torch.rand(len(rows), device=device, generator=generator) < .2
            active = torch.nonzero(chosen).flatten()
            valid[active, targets[active]+1] = False
            mask[active, targets[active]] = False
            targets[active] = 16
        stored = packet['patches'][ids.clamp_min(0)]
        patches, coords = prepare_patches(stored, coords, valid, augment, generator)
        return patches, coords, valid, mask, targets
    @torch.inference_mode()
    def evaluate(model, packet, missing=False):
        if model is not None:
            model.eval()
        scores, targets, movie_ids = [], [], []
        for first in range(0, len(packet['targets']), 12):
            rows = torch.arange(first, min(first+12, len(packet['targets'])), device=device)
            x, coord, valid, mask, target = batch(packet, rows, drop=missing)
            if model is None:
                distance2 = ((coord[:, 1:]-coord[:, :1])/10).square().sum(-1)
                raw = torch.cat((-2*distance2, torch.full_like(distance2[:, :1], -8)), -1)
                raw = raw.masked_fill(~torch.cat((valid[:, 1:], valid[:, :1]), -1), -1e4)
            else:
                with torch.autocast('cuda', dtype=torch.float16):
                    raw = model(x, coord, valid)
            raw = mask_scores(raw, mask).float()
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite validation logits')
            scores.append(raw.cpu()); targets.append(target.cpu()); movie_ids.append(packet['movie_ids'][rows].cpu())
        score, target, mi = torch.cat(scores), torch.cat(targets), torch.cat(movie_ids)
        loss = F.cross_entropy(score, target, reduction='none')
        prediction = score.argmax(-1); correct = prediction == target
        hard = (torch.softmax(score, -1).max(-1).values < .99)
        return dict(nll=float(loss.mean()), correct=int(correct.sum()), total=len(target),
                    present=int((target < 16).sum()), null=int((target == 16).sum()),
                    accuracy=float(correct.float().mean()), low_confidence=int(hard.sum()),
                    per_movie={name: dict(nll=float(loss[mi == i].mean()), correct=int(correct[mi == i].sum()),
                                         total=int((mi == i).sum())) for i, name in enumerate(packet['movie_names'])})
    summaries, smoke_timings = [], []
    model_pairs = [(e, f) for e in contract['source_embryos'] for f in contract['families']]
    if args.mode == 'smoke':
        model_pairs = model_pairs[:2]
    for index, (embryo, family) in enumerate(model_pairs):
        if stop_requested.is_set():
            result['status'] = 'yielded_before_member'; break
        member_start = time.monotonic()
        train, selection = data(embryo, 'optimization'), data(embryo, 'selection')
        seed = contract['seed'] + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed+971)
        model = VisualCorrespondence(family).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
        scaler = torch.amp.GradScaler('cuda', init_scale=2048.)
        member = args.output/f'{embryo}-{family}'; member.mkdir()
        baseline = evaluate(None, selection); missing_baseline = evaluate(None, selection, True)
        initial = evaluate(model, selection)
        if abs(initial['nll'] - baseline['nll']) > 1e-4:
            raise ValueError('Zero residual must reproduce distance prior')
        parameters = sum(p.numel() for p in model.parameters())
        summary = dict(embryo=embryo, family=family, parameters=parameters, seed=seed,
                       baseline=baseline, missing_baseline=missing_baseline, initial=initial,
                       optimization_movies=train['movie_names'], selection_movies=selection['movie_names'], history=[])
        best, best_step, optimizer_steps = float('inf'), 0, 0
        max_steps = 20 if args.mode == 'smoke' else contract['max_steps_per_member']
        train_tick = time.monotonic()
        for step in range(1, max_steps+1):
            model.train()
            rows = torch.multinomial(train['sampling_weights'], contract['batch_size'], replacement=True, generator=generator)
            x, coords, valid, mask, targets = batch(train, rows, generator, True, True)
            lr = 2e-6 + .5*(2e-4-2e-6)*(1+math.cos(math.pi*(step-1)/max_steps))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                scores = mask_scores(model(x, coords, valid), mask)
                loss = F.cross_entropy(scores.float(), targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite optimization loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            scale_before = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= scale_before:
                optimizer_steps += 1
            if not math.isfinite(float(norm)) and scaler.get_scale() >= scale_before:
                raise ValueError('Nonfinite unhandled gradient')
            checkpoint_due = step % contract['validation_every'] == 0 or step == max_steps
            timed_out = time.monotonic()-member_start > contract['max_seconds_per_member']-60
            if checkpoint_due or timed_out or stop_requested.is_set():
                metrics = evaluate(model, selection)
                row = dict(step=step, loss=float(loss), selection=metrics,
                           seconds=time.monotonic()-member_start, optimizer_steps=optimizer_steps)
                summary['history'].append(row)
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                if metrics['nll'] < best:
                    best, best_step = metrics['nll'], step
                    temporary = member/'best.partial'
                    torch.save(dict(state_dict=state, family=family, embryo=embryo, seed=seed,
                                    step=step, selection=metrics, bundle_sha256=args.bundle_sha256), temporary)
                    temporary.replace(member/'best.pt')
                temporary = args.output/'resume.partial'
                torch.save(dict(state_dict=state, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                family=family, embryo=embryo, step=step, seed=seed,
                                sampling_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all(), bundle_sha256=args.bundle_sha256), temporary)
                temporary.replace(args.output/'resume.pt')
                atomic_json(member/'history.json', summary)
                print(json.dumps(dict(member=f'{embryo}-{family}', step=step, parameters=parameters,
                     loss=row['loss'], validation_nll=metrics['nll'], baseline_nll=baseline['nll'],
                     correct=metrics['correct'], total=metrics['total'], seconds=row['seconds'])), flush=True)
            if timed_out or stop_requested.is_set():
                break
        elapsed = time.monotonic()-train_tick
        if optimizer_steps < max(1, int(step*.8)):
            raise ValueError('Too few actual optimizer updates')
        saved = torch.load(member/'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(saved['state_dict'], strict=True)
        selected = evaluate(model, selection)
        if abs(selected['nll']-saved['selection']['nll']) > 1e-6:
            raise ValueError('Checkpoint replay differs')
        missing = evaluate(model, selection, True)
        summary.update(status='trained_not_submission_ready', steps=step, best_step=best_step,
                       selected=selected, missing=missing, checkpoint_sha256=sha(member/'best.pt'),
                       optimizer_steps=optimizer_steps, seconds=time.monotonic()-member_start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), checkpoint_reload_passed=True,
                       source_screen_passed=selected['nll'] <= .98*baseline['nll'] and selected['correct'] >= baseline['correct']
                           and missing['nll'] <= missing_baseline['nll'],
                       full_movie_metric_gate_passed=False)
        atomic_json(member/'result.json', summary); summaries.append(summary)
        smoke_timings.append(elapsed/max_steps)
        del model, optimizer, scaler, saved; gc.collect(); torch.cuda.empty_cache()
        # Only one embryo's resident training set is needed at a time.
        if index % 2 == 1:
            data_cache.clear(); del train, selection; gc.collect(); torch.cuda.empty_cache()
        if stop_requested.is_set():
            result['status'] = 'yielded_with_checkpoint'; break
    stop_requested.set()
    result.update(members=summaries, authorized_for_submission=False,
                  complete_movie_scoring_performed=False, competition_submission_performed=False)
    if 'status' not in result:
        result['status'] = 'functionality_passed' if args.mode == 'smoke' else 'training_completed_requires_validation'
    if args.mode == 'smoke':
        result['estimated_four_member_seconds'] = 2*sum(smoke_timings)*contract['max_steps_per_member']*1.3 + 240


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    parser.add_argument('--smoke-proof', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(run_id='visual-correspondence-ensemble-v1', mode=args.mode, bundle_sha256=args.bundle_sha256)
    limit = 900 if args.mode == 'smoke' else 10080
    def hard_stop():
        result.update(status='hard_stop', elapsed_seconds=time.monotonic()-start)
        atomic_json(args.output/'result.json', result); os._exit(124)
    timer = threading.Timer(limit, hard_stop); timer.daemon = True; timer.start()
    try:
        main(args, result)
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        timer.cancel(); result['elapsed_seconds'] = time.monotonic()-start
        atomic_json(args.output/'result.json', result)
        print(json.dumps(dict(status=result['status'], seconds=result['elapsed_seconds'])), flush=True)
