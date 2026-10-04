"""Launch the frozen runtime-only test after live-job and quota checks."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.submission_sharding import validate_submission_kernel_metadata


def verify_submission_terminals(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or not {'ref', 'status'}.issubset(rows[0]):
        raise ValueError('Unrecognized submission inventory')
    for row in rows:
        if row['status'] not in ('SubmissionStatus.COMPLETE', 'SubmissionStatus.ERROR'):
            raise ValueError('Outstanding or unknown submission status; preserve its reservation')
    return rows


def main():
    receipt = ROOT / 'reports/experiments/trajectory-overlap-cache-kaggle-v1-launch.json'
    if receipt.exists():
        raise ValueError('Launch already attempted; inspect remote version, do not blindly retry')
    stage = ROOT / '.biohub/staging/biohub-trajectory-overlap-cache-acceptance-v1'
    build = json.loads((ROOT / 'reports/experiments/trajectory-overlap-cache-kaggle-v1-build.json').read_text())
    metadata = json.loads((stage / 'kernel-metadata.json').read_text())
    validate_submission_kernel_metadata(metadata)
    assert metadata['id'] == build['kernel'] == 'indarkarhana/biohub-trajectory-overlap-cache-acceptance'
    for name, key in ((metadata['code_file'], 'notebook_sha256'), ('derived-contract.json', 'contract_sha256'), ('overlays.json', 'overlay_sha256')):
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == build[key]
    assert build['acceptance_wall_cap_seconds'] == 3600 and build['required_gpus'] == 2
    # An empty device list alone is insufficient: also require the known research
    # job to have a terminal artifact. No process is terminated by this launcher.
    code = '''import json,subprocess
from pathlib import Path
p=Path('/dev/shm/biohub-dense-warp-movie-v1-full/result.json')
r=json.loads(p.read_text())
assert r['status'] in ('complete_prelabel_predictions','failed')
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout
assert not gpu.strip(), 'Antelume GPU occupied; sequential gate'
print(json.dumps(dict(research_status=r['status'],gpu_processes=[])))
'''
    check = subprocess.run(['ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=13.220.240.128',
        'ubuntu@3.226.249.134', '/home/ubuntu/venv/bin/python -'], input=code, text=True,
        capture_output=True, check=True, timeout=45)
    sequential = json.loads(check.stdout)
    kaggle = str(Path(sys.executable).parent / 'Scripts/kaggle.exe')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')

    def run(args):
        return subprocess.run([kaggle, *args], capture_output=True, text=True, encoding='utf-8',
                              env=env, check=True, timeout=45).stdout

    submissions = verify_submission_terminals(run(['competitions', 'submissions', '-c', 'biohub-cell-tracking-during-development', '--csv']))
    quota = json.loads(run(['quota', '--format', 'json']))
    gpu = [r for r in quota if r['resource'] == 'GPU']
    assert len(gpu) == 1
    remaining = float(gpu[0]['remaining'].removesuffix('h'))
    if remaining - 2 < 8:
        raise ValueError('Two-device one-hour cap would breach eight-hour reserve')
    result = dict(status='launch_requested', utc=datetime.now(timezone.utc).isoformat(),
                  kernel=metadata['id'], notebook_sha256=build['notebook_sha256'],
                  contract_sha256=build['contract_sha256'], quota=quota, sequential=sequential,
                  submission_statuses=[dict(ref=r['ref'], status=r['status']) for r in submissions],
                  worst_case_reserved_gpu_hours=2, reserve_hours=8, submission_performed=False)
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    try:
        response = subprocess.run([kaggle, 'kernels', 'push', '-p', '.', '-t', '3600'], cwd=stage,
            capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
        result.update(status='push_returned', exit_code=response.returncode,
                      stdout=response.stdout, stderr=response.stderr)
        if response.returncode:
            raise RuntimeError('Push failed; inspect receipt and remote state before retry')
    finally:
        receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
