"""One-shot, fresh-quota-gated launch of the offline two-T4 acceptance notebook."""
import json
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.submission_sharding import validate_submission_kernel_metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bootstrap-revision',type=int,choices=(2,3),default=2)
    parser.add_argument('--overlap',action='store_true')
    args=parser.parse_args(); revision=args.bootstrap_revision
    receipt_path=(ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-launch.json' if args.overlap
                  else ROOT/f'reports/experiments/trajectory-kaggle-acceptance-v{revision}-launch.json')
    if receipt_path.exists(): raise ValueError('Launch already attempted; inspect version before any retry')
    build=json.loads((ROOT/'reports/experiments/trajectory-kaggle-runtime-v2-build.json').read_text())
    if revision==3 and not args.overlap:
        repair=json.loads((ROOT/'reports/experiments/trajectory-kaggle-bootstrap-v3-build.json').read_text())
        if repair['contract_sha256']!=build['contract_sha256']:raise ValueError('Runtime changed during path-only fix')
        build['notebook_sha256']=repair['notebook_sha256']
    folder=ROOT/f'.biohub/staging/biohub-trajectory-motion-acceptance-v{revision}'
    expected_kernel='indarkarhana/biohub-trajectory-motion-acceptance'
    if args.overlap:
        build=json.loads((ROOT/'reports/experiments/trajectory-overlap-kaggle-v1-build.json').read_text())
        folder=ROOT/'.biohub/staging/biohub-trajectory-overlap-acceptance-v1'
        expected_kernel='indarkarhana/biohub-trajectory-overlap-acceptance'
    metadata=json.loads((folder/'kernel-metadata.json').read_text())
    validate_submission_kernel_metadata(metadata)
    if (metadata['id']!=expected_kernel
            or sha(folder/metadata['code_file'])!=build['notebook_sha256']):
        raise ValueError('Frozen kernel identity changed')
    kaggle=str(Path(sys.executable).parent/'Scripts/kaggle.exe')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    if revision==3 and not args.overlap:
        status=subprocess.run([kaggle,'kernels','status',metadata['id']],capture_output=True,text=True,
                              encoding='utf-8',check=True,timeout=45,env=env)
        if 'KernelWorkerStatus.ERROR' not in status.stdout:
            raise ValueError('Previous attempt must have terminated before replacement')
    quota=subprocess.run([kaggle,'quota','--format','json'],capture_output=True,text=True,
                         encoding='utf-8',check=True,timeout=45,env=env)
    rows=json.loads(quota.stdout); gpu=[r for r in rows if r['resource']=='GPU']
    if len(gpu)!=1: raise ValueError('Ambiguous GPU quota')
    remaining=float(gpu[0]['remaining'].removesuffix('h'))
    # Conservative reservation: two physical devices times the one-hour cap.
    if remaining-2 < 8: raise ValueError('Eight-hour GPU reserve would be violated')
    result=dict(status='launch_requested',utc=datetime.now(timezone.utc).isoformat(),
                kernel=metadata['id'],notebook_sha256=build['notebook_sha256'],
                contract_sha256=build['contract_sha256'],quota=rows,
                worst_case_reserved_gpu_hours=2,notebook_wall_cap_seconds=3600,
                reserve_hours=8,submission_performed=False)
    receipt_path.write_text(json.dumps(result,indent=2))
    try:
        response=subprocess.run([kaggle,'kernels','push','-p','.','-t','3600'],cwd=folder,
            capture_output=True,text=True,encoding='utf-8',timeout=120,env=env)
        result.update(status='push_returned',exit_code=response.returncode,
                      stdout=response.stdout,stderr=response.stderr)
        print(response.stdout,flush=True);print(response.stderr,flush=True)
        if response.returncode!=0: raise RuntimeError('Kaggle push failed; do not blind retry')
    finally:
        receipt_path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
