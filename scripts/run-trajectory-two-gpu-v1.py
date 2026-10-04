"""Launch exactly two isolated inference workers, then validate/merge all movies."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from submission_sharding import (build_movie_shards, visible_cuda_tokens,
    worker_environment, validate_shard_outputs, terminate_and_reap_processes,
    validate_inference_hard_stop, shard_plan_sha256)
from trajectory_runtime_v1 import (verify_bundle, image_metadata, validate_movie_ids,
                                    assemble_csv, sha)


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'images', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--contract-sha256', required=True)
    p.add_argument('--movies-json', type=Path)
    p.add_argument('--mode', choices=('smoke', 'validation', 'production'), required=True)
    p.add_argument('--wall-cap-seconds', type=int, required=True)
    p.add_argument('--smoke-proof', type=Path)
    args = p.parse_args()
    cap = validate_inference_hard_stop(args.wall_cap_seconds)
    started = time.monotonic()
    verify_bundle(args.bundle, args.contract_sha256)
    if args.mode != 'smoke':
        if not args.smoke_proof:
            raise ValueError('Same-contract two-GPU smoke required')
        smoke = json.loads(args.smoke_proof.read_text())
        if (smoke.get('status') != 'complete' or smoke.get('mode') != 'smoke'
                or smoke.get('contract_sha256') != args.contract_sha256
                or smoke.get('gpu_count') != 2):
            raise ValueError('Invalid smoke proof')
    if args.mode == 'production' and args.movies_json:
        raise ValueError('Production must discover every test movie, no subset permitted')
    movies = (json.loads(args.movies_json.read_text()) if args.movies_json
              else sorted(path.stem for path in args.images.glob('*.zarr')))
    validate_movie_ids(movies)
    metadata = {m: image_metadata(args.images / (m+'.zarr')) for m in movies}
    frames = {m: (8 if args.mode == 'smoke' else value.image_shape[0]) for m,value in metadata.items()}
    # Cheap fail-fast inventory, no truth files and no whole-image loading.
    weights = {}
    for movie in movies:
        chunks = [args.images / (movie+'.zarr') / '0/c' / str(t) / '0/0/0' for t in range(frames[movie])]
        if any(not c.is_file() or c.stat().st_size == 0 for c in chunks):
            raise ValueError(f'Missing image chunks: {movie}')
        weights[movie] = sum(c.stat().st_size for c in chunks)
    import torch
    devices = torch.cuda.device_count()
    if devices != 2:
        raise ValueError(f'Exactly two CUDA devices required, found {devices}')
    tokens = visible_cuda_tokens(detected_devices=devices)
    shards = build_movie_shards(movies, tokens, movie_weights=weights)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', mode=args.mode, gpu_count=devices,
                   torch_version=torch.__version__, contract_sha256=args.contract_sha256,
                   shard_plan_sha256=shard_plan_sha256(shards),
                   shards=[asdict(s) for s in shards], ground_truth_opened=False,
                   submission_created=False, authorized_for_submission=False)
    (args.output / 'plan.json').write_text(json.dumps(receipt, indent=2))
    processes, logs = [], []
    try:
        for shard in shards:
            movie_file = args.output / f'movies-{shard.shard_index}.json'
            movie_file.write_text(json.dumps(shard.movie_ids))
            worker_out = args.output / f'shard-{shard.shard_index}'
            log = (args.output / f'shard-{shard.shard_index}.log').open('x')
            logs.append(log)
            env = worker_environment(shard)
            env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       POLARS_MAX_THREADS='2', PYTHONUNBUFFERED='1')
            remaining = int(cap - (time.monotonic()-started))
            if remaining < 30:
                raise TimeoutError('No remaining launch budget')
            command = [sys.executable, str(args.bundle / 'portable-worker.py'),
                       '--bundle', str(args.bundle), '--images', str(args.images),
                       '--output', str(worker_out), '--movies-json', str(movie_file),
                       '--contract-sha256', args.contract_sha256,
                       '--mode', 'smoke' if args.mode == 'smoke' else 'full',
                       '--wall-cap-seconds', str(remaining)]
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        last_notice = -30
        while any(process.poll() is None for process in processes):
            elapsed = time.monotonic()-started
            if elapsed >= cap:
                raise TimeoutError('Two-GPU inference hard stop reached')
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('GPU worker failed; preserving logs, no partial submission')
            if elapsed-last_notice >= 30:
                print(json.dumps(dict(event='workers_running', elapsed_seconds=elapsed,
                                      exit_codes=[p.poll() for p in processes])), flush=True)
                last_notice = elapsed
            time.sleep(2)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError('Worker failed')
        observed, outputs, worker_receipts = {}, {}, {}
        for shard in shards:
            folder = args.output / f'shard-{shard.shard_index}'
            result = json.loads((folder / 'result.json').read_text())
            if (result.get('status') not in ('functionality_passed', 'complete_prelabel_predictions')
                    or result.get('inputs_unchanged') is not True
                    or result.get('contract_sha256') != args.contract_sha256):
                raise ValueError('Incomplete or changed worker output')
            observed[shard.shard_index] = list(result['movies'])
            worker_receipts[str(shard.shard_index)] = result
            for movie, arms in result['movies'].items():
                output = folder / f'{movie}-original/repaired-prediction.json'
                if (sha(output) != arms['original']['repaired_sha256']
                        or arms['original']['frames'] != frames[movie]):
                    raise ValueError('Changed graph or missing frames')
                outputs[movie] = output
        validate_shard_outputs(shards, observed)
        if time.monotonic()-started >= cap:
            raise TimeoutError('No finalization budget')
        filename = 'submission.csv' if args.mode == 'production' else args.mode+'-predictions.csv'
        csv_receipt = assemble_csv(outputs, frames, args.output / filename)
        receipt.update(status='complete', csv=csv_receipt, workers=worker_receipts,
                       submission_created=args.mode == 'production')
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        terminate_and_reap_processes(processes)
        for log in logs:
            log.close()
        receipt['elapsed_seconds'] = time.monotonic()-started
        (args.output / 'result.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps({k:v for k,v in receipt.items() if k != 'workers'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
