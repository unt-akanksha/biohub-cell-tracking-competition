"""One bounded experiment: two FP32 movie workers share one otherwise idle GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from trajectory_runtime_v1 import verify_bundle, sha, validate_graph

MOVIES = (('44b6_12dfb391', '6bba_062c8d37'),
          ('44b6_267148e4', '6bba_07e24132'))


def gpu_pids():
    query = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                           capture_output=True,text=True,check=True,timeout=15).stdout
    return {int(v.strip()) for v in query.splitlines() if v.strip()}


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle','images','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--contract-sha256',required=True)
    p.add_argument('--mode',choices=('smoke','full'),required=True)
    p.add_argument('--smoke-proof',type=Path)
    args = p.parse_args()
    start = time.monotonic()
    cap = 600 if args.mode == 'smoke' else 1800
    contract = verify_bundle(args.bundle,args.contract_sha256)
    if contract['run_id'] != 'trajectory-overlap-v1':
        raise ValueError('Wrong experiment')
    if args.mode == 'full':
        smoke = json.loads(args.smoke_proof.read_text())
        if (smoke['status']!='functionality_passed' or smoke['contract_sha256']!=args.contract_sha256
                or smoke['mode']!='smoke' or not smoke['inputs_unchanged']):
            raise ValueError('Same-contract overlap smoke required')
    if gpu_pids():
        raise ValueError('Foreign GPU workload present; no sharing without a free device')
    args.output.mkdir(exist_ok=False)
    result = dict(run_id='trajectory-overlap-v1',status='running',mode=args.mode,
                  contract_sha256=args.contract_sha256,ground_truth_opened=False,
                  authorized_for_submission=False,movies={},workers={},gpu_samples=[],
                  gpu_count=1,concurrent_workers=2,total_cpu_threads=2,
                  aggregate_cuda_allocator_fraction=.70,inputs_unchanged=False)
    processes = []
    logs = []
    try:
        for index,assigned in enumerate(MOVIES):
            movies = assigned[:1] if args.mode == 'smoke' else assigned
            path = args.output/f'movies-{index}.json'
            path.write_text(json.dumps(movies))
            log = (args.output/f'shard-{index}.log').open('x');logs.append(log)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',POLARS_MAX_THREADS='1',
                       PYTHONUNBUFFERED='1',PYTHONPATH=str(args.bundle))
            command = [sys.executable,str(args.bundle/'portable-worker.py'),
                       '--bundle',str(args.bundle),'--images',str(args.images),
                       '--output',str(args.output/f'shard-{index}'),'--movies-json',str(path),
                       '--contract-sha256',args.contract_sha256,'--mode',args.mode,
                       '--wall-cap-seconds',str(cap-20)]
            processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
        sample_at = 0
        while any(p.poll() is None for p in processes):
            elapsed = time.monotonic()-start
            if elapsed > cap:
                raise TimeoutError('Overlap walltime limit')
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('Owned worker failed')
            if elapsed >= sample_at:
                foreign = gpu_pids()-{p.pid for p in processes}
                if foreign:
                    raise RuntimeError('Foreign GPU workload appeared; yielding owned workers')
                sample = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                                         '--format=csv,noheader,nounits'],capture_output=True,
                                        text=True,check=True,timeout=15).stdout.strip()
                record = dict(elapsed_seconds=elapsed,utilization_and_mib=sample)
                result['gpu_samples'].append(record)
                print(json.dumps(dict(event='overlap_progress',**record)),flush=True)
                sample_at = elapsed+30
            time.sleep(2)
        if any(p.returncode!=0 for p in processes):
            raise RuntimeError('Owned worker failed')
        for index,assigned in enumerate(MOVIES):
            folder = args.output/f'shard-{index}'
            worker = json.loads((folder/'result.json').read_text())
            expected = set(assigned[:1] if args.mode=='smoke' else assigned)
            if (worker['status'] not in ('functionality_passed','complete_prelabel_predictions')
                    or worker['contract_sha256']!=args.contract_sha256 or not worker['inputs_unchanged']
                    or set(worker['movies'])!=expected or worker['ground_truth_opened']):
                raise ValueError('Incomplete or changed worker')
            result['workers'][str(index)]=worker
            for stem,arms in worker['movies'].items():
                for filename,field in (('prediction.json','prediction_sha256'),
                                       ('repaired-prediction.json','repaired_sha256')):
                    path=folder/(stem+'-original')/filename
                    if sha(path)!=arms['original'][field]:
                        raise ValueError('Worker output changed')
                    validate_graph(json.loads(path.read_text()),8 if args.mode=='smoke' else 100)
                result['movies'][stem]=dict(**arms,shard=index)
        verify_bundle(args.bundle,args.contract_sha256)
        result.update(status='functionality_passed' if args.mode=='smoke' else 'complete_prelabel_predictions',
                      inputs_unchanged=True,quality_gain_established=False)
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        for log in logs:log.close()
        result['elapsed_seconds']=time.monotonic()-start
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('workers','gpu_samples')},indent=2),flush=True)


if __name__=='__main__':
    main()
