"""One-shot detached source collection, explicit roots, no foreign job cleanup."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SSH = ['-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=13.220.240.128']
HOST = 'ubuntu@3.226.249.134'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--mode', choices=('smoke', 'full'), required=True)
    p.add_argument('--remote-root', required=True)
    args = p.parse_args()
    assert re.fullmatch(r'/dev/shm/biohub-event-source-v1-b' + str(args.batch) + r'\.[A-Za-z0-9]{6}', args.remote_root)
    name = 'trajectory-event-source-v1-b' + str(args.batch)
    receipt = ROOT / 'reports/experiments' / (name + '-' + args.mode + '-launch.json')
    assert not receipt.exists(), 'Previously attempted; inspect actual state before any retry'
    build_path = ROOT / 'reports/experiments' / (name + '-build.json')
    build = json.loads(build_path.read_text())
    assert build['batch'] == args.batch and not build['authorized_for_submission']
    proof_sha = None
    if args.mode == 'full':
        proof = ROOT / '.biohub/cache' / (name + '-smoke-output/result.json')
        smoke = json.loads(proof.read_text())
        backup = json.loads((ROOT / 'reports/experiments' / (name + '-smoke-harvest.json')).read_text())
        assert backup['status'] == 'verified_backup'
        assert smoke['status'] == 'functionality_passed' and smoke['inputs_unchanged']
        assert smoke['contract_sha256'] == build['contract_sha256']
        assert list(smoke['movies']) == build['movies'][:1]
        assert all(a['original']['frames'] == 8 for a in smoke['movies'].values())
        proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest()
        assert any(r['path'] == 'result.json' and r['sha256'] == proof_sha for r in backup['records'])
    payload = dict(root=args.remote_root, name=name, mode=args.mode, build=build, smoke_sha=proof_sha)
    code = '''import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
payload = ''' + repr(payload) + '''
root = Path(payload['root']); build = payload['build']; mode = payload['mode']
assert root.is_dir() and not root.is_symlink() and root.resolve() == root and root.parent == Path('/dev/shm')
out = root/mode
assert not out.exists(), 'Output already exists; inspect rather than restart'
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
 try: command = (proc/'cmdline').read_bytes()
 except (FileNotFoundError,ProcessLookupError,PermissionError): continue
 assert b'run-trajectory-division-full-movie-v1.py' not in command, 'Other source runner PID '+proc.name
query = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
assert not query.stdout.strip(), 'GPU occupied; do not disrupt other projects'
memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(memory['MemAvailable'].split()[0])*1024 > 3*1024**3, 'Insufficient available host RAM'
bundle = root/(payload['name']+'-bundle')
if mode == 'smoke':
 archive = root/(payload['name']+'-bundle.tar')
 assert hashlib.sha256(archive.read_bytes()).hexdigest() == build['archive_sha256'], 'Incomplete or changed upload'
 assert not bundle.exists()
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():
   path = Path(member.name)
   assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == bundle.name
   assert member.isdir() or member.isfile()
  tar.extractall(root,filter='data')
else:
 assert hashlib.sha256((root/'smoke/result.json').read_bytes()).hexdigest() == payload['smoke_sha']
contract = bundle/'CONTRACT.json'
assert hashlib.sha256(contract.read_bytes()).hexdigest() == build['contract_sha256']
assert hashlib.sha256((root/'images/IMAGE_MANIFEST.json').read_bytes()).hexdigest() == build['images_sha256']
for name,digest in json.loads(contract.read_text())['bundle_sha256'].items():
 assert Path(name).name == name and hashlib.sha256((bundle/name).read_bytes()).hexdigest() == digest
command = ['/home/ubuntu/venv/bin/python',str(bundle/'run-trajectory-division-full-movie-v1.py'),
 '--bundle',str(bundle),'--images',str(root/'images'),'--contract',str(contract),
 '--contract-sha256',build['contract_sha256'],'--mode',mode,'--output',str(out)]
if mode == 'full': command += ['--smoke-proof',str(root/'smoke/result.json')]
env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',POLARS_MAX_THREADS='2')
with (root/(mode+'.log')).open('x') as log:
 process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=bundle,start_new_session=True)
print(json.dumps(dict(status='remote_process_started',pid=process.pid,root=str(root),mode=mode,
 log=str(root/(mode+'.log')),contract_sha256=build['contract_sha256'],started_epoch=time.time(),
 wall_cap_seconds=600 if mode=='smoke' else 2700)))
'''
    record = dict(status='launch_requested', utc=datetime.now(timezone.utc).isoformat(),
                  batch=args.batch, mode=args.mode, remote_root=args.remote_root,
                  build_sha256=hashlib.sha256(build_path.read_bytes()).hexdigest(),
                  contract_sha256=build['contract_sha256'], foreign_jobs_untouched=True,
                  no_kaggle_experiment_launched=True, frozen_submission_untouched=True)
    receipt.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    response = subprocess.run(['ssh', *SSH, HOST, '/home/ubuntu/venv/bin/python -'],
                              input=code, capture_output=True, text=True, timeout=60)
    record['exit_code'] = response.returncode
    if response.returncode:
        record.update(status='launch_outcome_requires_inspection', stderr=response.stderr)
    else:
        record.update(status='remote_process_started', remote=json.loads(response.stdout))
    receipt.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    if response.returncode:
        raise SystemExit(response.returncode)


if __name__ == '__main__':
    main()
