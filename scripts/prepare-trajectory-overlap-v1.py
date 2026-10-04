"""Freeze same FP32 inference math; only worker scheduling/resource caps change."""
import ast
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_portable_adapter_v1 import portable_source,replace_once


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exact-tta-cache',action='store_true')
    args=parser.parse_args()
    run_id='trajectory-overlap-cache-v1' if args.exact_tta_cache else 'trajectory-overlap-v1'
    source=ROOT/'.biohub/cache/trajectory-division-full-movie-v1-bundle'
    if sha(source/'CONTRACT.json')!='7d4d1bc9f81f37fc2a8d23bd2449fc53f99bfb23eff39d02460b884661f8bdd1':
        raise ValueError('Frozen FP32 baseline changed')
    contract=json.loads((source/'CONTRACT.json').read_text())
    out=ROOT/f'.biohub/cache/{run_id}-bundle';out.mkdir(exist_ok=False)
    for name,digest in contract['bundle_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Baseline input changed')
        shutil.copyfile(source/name,out/name)
    code=portable_source((source/'run-trajectory-division-full-movie-v1.py').read_text())
    # Model initialization, arithmetic and expensive prediction/ILP/repair block
    # remain exactly the accepted FP32 implementation. Resource controls only.
    for old,new in (("OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'",
                     "OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'"),
                    ("POLARS_MAX_THREADS='2'","POLARS_MAX_THREADS='1'"),
                    ('torch.set_num_threads(2)','torch.set_num_threads(1)'),
                    ('torch.cuda.set_per_process_memory_fraction(.70)',
                     'torch.cuda.set_per_process_memory_fraction(.35)')):
        code=replace_once(code,old,new)
    ast.parse(code);(out/'portable-worker.py').write_text(code)
    shutil.copyfile(ROOT/'research/trajectory_runtime_v1.py',out/'trajectory_runtime_v1.py')
    shutil.copyfile(ROOT/'scripts/run-trajectory-overlap-v1.py',out/'run-trajectory-overlap-v1.py')
    if args.exact_tta_cache:
        from research.trajectory_exact_tta_cache_v1 import cached_predictor
        path=out/'public-predictor-original.py'
        cached=cached_predictor(path.read_text());ast.parse(cached);path.write_text(cached)
        controller=out/'run-trajectory-overlap-v1.py'
        controller.write_text(controller.read_text().replace('trajectory-overlap-v1',run_id))
    contract.update(run_id=run_id,ground_truth_included=False,
                    public_prediction_tables_included=False,authorized_for_submission=False,
                    scheduling='Two one-thread FP32 movie workers on one idle A10G; 35% allocator each',
                    baseline_contract_sha256=sha(source/'CONTRACT.json'),
                    no_precision_or_model_math_changes=True,
                    exact_duplicate_tta_encode_cache=args.exact_tta_cache)
    contract['bundle_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'CONTRACT.json').write_text(json.dumps(contract,indent=2))
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(out.iterdir()):tar.add(path,arcname=path.name,recursive=False)
    result=dict(status='staged',contract_sha256=sha(out/'CONTRACT.json'),archive_sha256=sha(archive),
                archive_bytes=archive.stat().st_size,full_run_cap_seconds=1800,
                smoke_required=True,fp32_baseline_preserved=True,submission_performed=False)
    (ROOT/f'reports/experiments/{run_id}-build.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
