"""Back up completed owned visual-model artifacts with per-file SHA verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/tmp/biohub-image-context-v2.ScdSdY'
HOST = 'ubuntu@3.226.249.134'
SSH_OPTIONS = ['-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'HostKeyAlias=13.220.240.128']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'train'), default='train')
    args = parser.parse_args()
    start = time.monotonic()
    name = f'visual-correspondence-{args.mode}-v1-r2'
    code = f'''import hashlib,json
from pathlib import Path
root=Path({(REMOTE_ROOT+'/'+name)!r}).resolve(strict=True)
terminal=json.loads((root/'result.json').read_text())
assert terminal['status'] in ('functionality_passed','training_completed_requires_validation')
assert terminal['bundle_sha256']=='a3a59450791c4d6aef7e75f8d7e2a40c5a41f9dc1a4a14d0440f0a1f0e1712d0'
names=['result.json','resume.pt']
for m in terminal['members']:
 assert m['embryo'] in ('44b6','6bba') and m['family'] in ('resnet3d','token_transformer3d')
 prefix=m['embryo']+'-'+m['family']+'/'
 names.extend(prefix+n for n in ('best.pt','history.json','result.json'))
records=[]
for name in names:
 path=(root/name).resolve(strict=True)
 assert root in path.parents and path.is_file() and path.stat().st_size<512*1024**2
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 records.append(dict(path=name,bytes=path.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(root=str(root),status=terminal['status'],records=records)))
'''
    command = ['ssh', *SSH_OPTIONS, HOST, '/home/ubuntu/venv/bin/python -c '+shlex.quote(code)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    inventory = json.loads(completed.stdout)
    output = ROOT/'.biohub/cache'/f'{name}-output'
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 1.2*sum(r['bytes'] for r in inventory['records']):
        raise ValueError('Insufficient local backup space')
    def fetch(record):
        destination = output/record['path']
        if output.resolve() not in destination.resolve().parents:
            raise ValueError('Backup path escaped exact output folder')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha(destination) != record['sha256']:
                raise ValueError('Preserve changed existing backup')
            return record
        temporary = destination.with_suffix(destination.suffix+'.download')
        if temporary.exists():
            raise ValueError('Incomplete prior download requires explicit review')
        subprocess.run(['scp', *SSH_OPTIONS, HOST+':'+inventory['root']+'/'+record['path'], str(temporary)],
                       check=True, capture_output=True, timeout=180)
        if temporary.stat().st_size != record['bytes'] or sha(temporary) != record['sha256']:
            raise ValueError('Artifact transfer verification failed')
        temporary.replace(destination)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, inventory['records']))
    result = dict(status='verified_backup', output=str(output), records=records,
                  total_bytes=sum(r['bytes'] for r in records), elapsed_seconds=time.monotonic()-start,
                  remote_files_deleted=False)
    receipt = ROOT/'reports/experiments'/f'{name}-harvest.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['records'] != records:
            raise ValueError('Preserve differing prior backup receipt')
    else:
        receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
