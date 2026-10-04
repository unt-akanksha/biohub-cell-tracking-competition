"""Remove four redundant owned cloud copies only after exact local backup checks."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
receipt = ROOT/'reports/experiments/visual-correspondence-redundant-copies-cleanup-v1.json'
if receipt.exists():
    raise ValueError('Cleanup already has a receipt; do not repeat')
chosen = {
    'smoke': {'resume.pt', '6bba-resnet3d/best.pt', '6bba-token_transformer3d/best.pt'},
    'train': {'resume.pt'},
}
records = []
for mode, names in chosen.items():
    name = f'visual-correspondence-{mode}-v1-r2'
    backup = json.loads((ROOT/'reports/experiments'/f'{name}-harvest.json').read_text())
    if backup['status'] != 'verified_backup':
        raise ValueError('Verified completed backup required')
    for row in backup['records']:
        if row['path'] not in names:
            continue
        path = (ROOT/'.biohub/cache'/f'{name}-output'/row['path']).resolve(strict=True)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                h.update(block)
        if path.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
            raise ValueError('Local recovery copy failed SHA/size check')
        records.append(dict(relative=name+'/'+row['path'], local_recovery=str(path),
                            sha256=row['sha256'], bytes=row['bytes']))
if len(records) != 4:
    raise ValueError('Expected exactly four redundant copies')
code = '''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/tmp/biohub-image-context-v2.ScdSdY').resolve(strict=True)
records=INPUT_RECORDS
pids=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True,timeout=15).stdout.strip()
assert not pids, 'No live GPU work may be present during cleanup'
targets=[]
for r in records:
 unresolved=root/r['relative']; assert not unresolved.is_symlink()
 p=unresolved.resolve(strict=True)
 assert root in p.parents and p.is_file() and p.stat().st_size==r['bytes']
 assert p.name in ('resume.pt','best.pt')
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 assert h.hexdigest()==r['sha256']
 targets.append(p)
before=shutil.disk_usage(root).free
for p in targets:p.unlink()
print(json.dumps(dict(status='removed_verified_redundant_copies',records=records,
 recovered_bytes=sum(r['bytes'] for r in records),free_bytes_before=before,
 free_bytes_after=shutil.disk_usage(root).free,all_full_training_best_weights_retained=True,
 rsna_files_or_processes_touched=False)))
'''.replace('INPUT_RECORDS', repr(records))
command = ['ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'HostKeyAlias=13.220.240.128', 'ubuntu@3.226.249.134',
           '/home/ubuntu/venv/bin/python -c '+shlex.quote(code)]
response = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
result = json.loads(response.stdout)
receipt.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'records'}))
