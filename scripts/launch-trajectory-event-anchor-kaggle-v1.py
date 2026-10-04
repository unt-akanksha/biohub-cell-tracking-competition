"""One-shot acceptance launch with documented quota units and live-job checks."""
import argparse
import csv
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_quota_v1 import acceptance_budget,saved_kernel_complete
from research.submission_sharding import validate_submission_kernel_metadata


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    receipt=ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-launch.json'
    assert not receipt.exists(),'Launch attempted already: inspect remote version, never blindly retry'
    stage=ROOT/'.biohub/staging/biohub-event-anchor-acceptance-v1'
    build=read(ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-build.json')
    metadata=read(stage/'kernel-metadata.json');validate_submission_kernel_metadata(metadata)
    assert metadata['id']==build['kernel']=='indarkarhana/biohub-event-anchor-acceptance'
    assert metadata['machine_shape']=='NvidiaTeslaT4' and metadata['is_private'] and not metadata['enable_internet']
    for name,key in ((metadata['code_file'],'notebook_sha256'),('derived-contract.json','contract_sha256'),('overlays.json','overlay_sha256')):
        assert sha(stage/name)==build[key]
    assert build['acceptance_wall_cap_seconds']==3600 and build['required_gpus']==2
    portable=ROOT/'.biohub/cache/trajectory-event-anchor-portable-v1'
    assert sha(portable/'RESULT.json')==build['event_portable_proof_sha256']
    assert read(portable/'RESULT.json')['status']=='portable_inference_verified'
    policy_path=ROOT/'reports/experiments/trajectory-event-t4-quota-policy-v1.json';policy=read(policy_path)
    assert policy['status']=='direct_staff_t4_quota_clarification_verified'
    assert policy['comment_id']==1995818 and policy['comment_content_sha256']=='301a1e0f286674a4e343177493be3f193cb34ec294e31c06bdc7c43550e8af2c'
    assert policy['t4x2_quota_hours_per_notebook_hour']==1 and policy['pending_platform_cap_reserved_hours']==12
    confirmation=ROOT/'reports/experiments/trajectory-structured-v1-submission-confirmation.json'
    assert sha(confirmation)=='376223a49f27f04f758bb2cff92f1855583e6fff282f603c182f9aed51bc4c4c'
    confirmed=read(confirmation);assert confirmed['submission_ref']==56231458 and confirmed['kernel_version']==1
    controller_path=ROOT/'reports/experiments/trajectory-event-source-v1-controller-r2.json'
    controller=read(controller_path)
    if controller['status']!='complete':
        print(json.dumps(dict(status='waiting_for_existing_source_controller',active_batch=controller['active_batch'],
            phase=controller.get('phase'),kernel_pushed=False,check_only=args.check_only)),flush=True)
        return
    assert controller['through_batch']==7 and set(controller['completed'])==set(range(1,8))
    launch=read(ROOT/'reports/experiments/trajectory-event-source-v1-b7-full-launch.json')
    remote_root=launch['remote_root'];pid=int(launch['remote']['pid'])
    assert remote_root.startswith('/dev/shm/biohub-event-source-v1-b7.') and controller['remote_root']==remote_root
    code="""import json,subprocess
from pathlib import Path
root=Path(ROOT_VALUE)
r=json.loads((root/'full/result.json').read_text())
assert r['status']=='complete_prelabel_predictions' and r['contract_sha256']==CONTRACT_VALUE
p=Path('/proc')/str(PID_VALUE)/'cmdline'
assert not p.exists() or str(root).encode() not in p.read_bytes(),'Known Biohub process is still live'
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout
gpu_pids=[int(v.strip()) for v in gpu.splitlines() if v.strip()]
for gpu_pid in gpu_pids:
    proc=Path('/proc')/str(gpu_pid)
    try:
        cmd=(proc/'cmdline').read_bytes().decode(errors='replace')
        cwd=str((proc/'cwd').resolve())
    except FileNotFoundError:
        continue
    assert 'biohub-' not in cmd.lower() and 'biohub-' not in cwd.lower(),'Another Biohub GPU experiment is live'
print(json.dumps(dict(last_source_status=r['status'],last_source_pid=PID_VALUE,
    live_biohub_gpu_processes=[],foreign_gpu_processes_untouched=gpu_pids)))
""".replace('ROOT_VALUE',repr(remote_root)).replace('CONTRACT_VALUE',repr(launch['contract_sha256'])).replace('PID_VALUE',str(pid))
    check=subprocess.run(['ssh','-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes',
        '-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128',
        'ubuntu@3.226.249.134','/home/ubuntu/venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=45)
    sequential=json.loads(check.stdout)
    kaggle=str(Path(sys.executable).parent/'Scripts/kaggle.exe')
    assert Path(kaggle).exists(),'Use the base Python that owns the authenticated Kaggle CLI'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    def run(arguments):
        return subprocess.run([kaggle,*arguments],capture_output=True,text=True,encoding='utf-8',env=env,check=True,timeout=45).stdout
    prior_status=run(['kernels','status',confirmed['kernel']])
    assert saved_kernel_complete(prior_status,confirmed['kernel']),'Prior saved notebook is not terminal'
    submissions=list(csv.DictReader(io.StringIO(run(['competitions','submissions','-c','biohub-cell-tracking-during-development','--csv']))))
    quota=json.loads(run(['quota','--format','json']));gpu=[r for r in quota if r['resource']=='GPU'];assert len(gpu)==1
    assert gpu[0]['remaining'].endswith('h')
    budget=acceptance_budget(float(gpu[0]['remaining'][:-1]),submissions,{56231458:12.})
    result=dict(status='preflight_passed',utc=datetime.now(timezone.utc).isoformat(),kernel=metadata['id'],
        notebook_sha256=build['notebook_sha256'],contract_sha256=build['contract_sha256'],quota=quota,budget=budget,
        sequential=sequential,controller_sha256=sha(controller_path),quota_policy_sha256=sha(policy_path),
        submission_statuses=[dict(ref=r['ref'],status=r['status']) for r in submissions],
        quality_failures_preserved=True,submission_performed=False,kernel_pushed=False)
    if args.check_only:
        print(json.dumps(result),flush=True);return
    result['status']='launch_requested';receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    try:
        pushed=subprocess.run([kaggle,'kernels','push','-p','.', '-t','3600'],cwd=stage,
            capture_output=True,text=True,encoding='utf-8',env=env,timeout=120)
        result.update(status='push_returned',exit_code=pushed.returncode,stdout=pushed.stdout,stderr=pushed.stderr,
                      kernel_pushed=pushed.returncode==0)
        if pushed.returncode:raise RuntimeError('Push failed; inspect recorded remote state before another attempt')
    finally:
        receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
