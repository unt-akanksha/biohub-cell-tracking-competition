"""One-shot private production run; requires quality/headroom and fresh quota."""
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.submission_sharding import validate_submission_kernel_metadata


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--overlap',action='store_true')
    args=parser.parse_args()
    prefix='trajectory-overlap-production-v1' if args.overlap else 'trajectory-production-v1'
    kernel='indarkarhana/biohub-trajectory-overlap-candidate' if args.overlap else 'indarkarhana/biohub-trajectory-motion-candidate'
    target=ROOT/f'reports/experiments/{prefix}-launch.json'
    if target.exists():raise ValueError('Already attempted; inspect actual state before retry')
    build=json.loads((ROOT/f'reports/experiments/{prefix}-build.json').read_text())
    proof=ROOT/('reports/experiments/trajectory-overlap-kaggle-v1-result.json' if args.overlap
                else 'reports/experiments/trajectory-kaggle-acceptance-v2-result.json')
    if sha(proof)!=build['acceptance_sha256']:raise ValueError('Acceptance changed')
    acceptance=json.loads(proof.read_text())
    if (acceptance['status']!='acceptance_passed' or build['status']!='production_staged_not_launched'
            or not acceptance['comparison']['diagnostic_gate_passed']
            or not 0<build['runtime_projection_hours']<=8
            or build['runtime_budget_policy_revision']!=2
            or build['minimum_runtime_headroom_hours']!=2
            or build['inference_watchdog_seconds']!=36000):
        raise ValueError('Quality and declared headroom required')
    folder=ROOT/('.biohub/staging/biohub-trajectory-overlap-candidate-v1' if args.overlap
                 else '.biohub/staging/biohub-trajectory-motion-candidate-v1')
    metadata=json.loads((folder/'kernel-metadata.json').read_text())
    validate_submission_kernel_metadata(metadata)
    if (metadata['id']!=kernel
            or sha(folder/metadata['code_file'])!=build['notebook_sha256']):
        raise ValueError('Staged candidate changed')
    cli=str(Path(sys.executable).parent/'Scripts/kaggle.exe')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    status=subprocess.run([cli,'kernels','status','indarkarhana/biohub-trajectory-overlap-acceptance'],
                          capture_output=True,text=True,encoding='utf-8',check=True,timeout=45,env=env)
    if 'KernelWorkerStatus.COMPLETE' not in status.stdout:
        raise ValueError('Acceptance must have finished before production')
    response=subprocess.run([cli,'quota','--format','json'],capture_output=True,text=True,
                            encoding='utf-8',check=True,timeout=45,env=env)
    quota=json.loads(response.stdout);gpu=[r for r in quota if r['resource']=='GPU']
    if len(gpu)!=1 or float(gpu[0]['remaining'].removesuffix('h'))-20<8:
        raise ValueError('Conservative two-device ten-hour reservation violates eight-hour reserve')
    result=dict(status='launch_requested',utc=datetime.now(timezone.utc).isoformat(),
                kernel=metadata['id'],notebook_sha256=build['notebook_sha256'],
                contract_sha256=build['contract_sha256'],acceptance_sha256=sha(proof),
                quota=quota,worst_case_reserved_gpu_hours=20,reserve_hours=8,
                notebook_wall_cap_seconds=36000,submission_performed=False)
    target.write_text(json.dumps(result,indent=2))
    try:
        push=subprocess.run([cli,'kernels','push','-p','.','-t','36000'],cwd=folder,
                            capture_output=True,text=True,encoding='utf-8',timeout=120,env=env)
        result.update(status='push_returned',exit_code=push.returncode,stdout=push.stdout,stderr=push.stderr)
        if push.returncode!=0:raise RuntimeError('Push failed; inspect state, no blind retry')
    finally:
        target.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
