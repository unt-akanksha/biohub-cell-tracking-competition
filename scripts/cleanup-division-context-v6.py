"""Remove exact terminal v6 RAM copies only after local and remote hash checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RECOVERIES = (
    ('native-division-context-v6-harvest', 'native-division-context-v6-recovery-v1'),
    ('native-division-context-v6-training-harvest-r1', 'native-division-context-v6-training-recovery-r1'),
    ('native-division-context-v6-training-harvest-r2', 'native-division-context-v6-training-recovery-r2'),
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    receipt = ROOT / 'reports/experiments/native-division-context-v6-memory-cleanup.json'
    if receipt.exists():
        raise ValueError('Preserve completed cleanup receipt')
    records = []
    groups = set()
    for receipt_name, archive_name in RECOVERIES:
        backup = json.loads((ROOT / 'reports/experiments' / (receipt_name + '.json')).read_text())
        assert backup['status'] == 'verified_backup'
        for row in backup['records']:
            local_root = (ROOT / '.biohub/cache' / (row['group'] + '-output')).resolve(strict=True)
            local = (local_root / row['path']).resolve(strict=True)
            assert local.is_relative_to(local_root)
            assert local.stat().st_size == row['bytes'] and sha(local) == row['sha256']
            groups.add('/dev/shm/biohub-' + row['group'])
            records.append(dict(path='/dev/shm/biohub-' + row['group'] + '/' + row['path'],
                                bytes=row['bytes'], sha256=row['sha256'], local_recovery=str(local)))
        archive = ROOT / '.biohub/cache' / (archive_name + '.tar')
        assert sha(archive) == backup['archive_sha256']
        records.append(dict(path='/dev/shm/biohub-' + archive_name + '.tar',
                            bytes=archive.stat().st_size, sha256=backup['archive_sha256'],
                            local_recovery=str(archive)))
    code = '''import hashlib,json,subprocess,shutil,os,datetime
from pathlib import Path
''' + f'records={records!r}\ngroups={sorted(groups)!r}\nexecute={args.execute!r}\n' + '''
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.strip(), 'GPU occupied; preserve all files'
expected={r['path'] for r in records}
assert len(expected)==len(records)
directories=[]
for group in groups:
 root=Path(group)
 assert root.parent==Path('/dev/shm') and root.name.startswith('biohub-native-division-context-v6-')
 assert not root.is_symlink() and root.resolve(strict=True)==root
 assert json.loads((root/'RESULT.json').read_text())['status'] in ('failed','temporal_training_complete','context_smoke_passed','context_data_complete','temporal_model_smoke_passed')
 actual=set()
 for p in root.rglob('*'):
  assert not p.is_symlink() and p.resolve(strict=True).is_relative_to(root)
  if p.is_file():actual.add(str(p))
  elif p.is_dir():directories.append(p)
  else:raise ValueError('Unexpected filesystem object')
 assert actual=={p for p in expected if p.startswith(group+'/')}, 'Remote file set changed'
 directories.append(root)
for row in records:
 p=Path(row['path'])
 assert not p.is_symlink() and p.resolve(strict=True)==p and p.is_file()
 assert any(p.is_relative_to(Path(g)) for g in groups) or (p.parent==Path('/dev/shm') and p.name.startswith('biohub-native-division-context-v6-') and p.suffix=='.tar')
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
# Refuse any process holding one of these exact files or referencing a target tree.
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\\0')
  assert not any(any(arg.decode(errors='replace')==g or arg.decode(errors='replace').startswith(g+'/') for g in groups) for arg in cmd), 'Process references cleanup target'
  links=[proc/'cwd',*(proc/'fd').iterdir()]
  for link in links:
   try:target=os.readlink(link)
   except (FileNotFoundError,PermissionError):continue
   assert target not in expected and not any(target==g or target.startswith(g+'/') for g in groups), 'Target is in use'
 except (FileNotFoundError,PermissionError):continue
before=shutil.disk_usage('/dev/shm').free
if execute:
 for row in records:Path(row['path']).unlink()
 for p in sorted(directories,key=lambda x:len(x.parts),reverse=True):p.rmdir()
 assert all(not Path(r['path']).exists() for r in records)
result=dict(status='removed_verified_terminal_ram_copies' if execute else 'verified_dry_run',
 timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 recovered_bytes=sum(r['bytes'] for r in records) if execute else 0,
 planned_bytes=sum(r['bytes'] for r in records),files=len(records),groups=groups,
 free_before=before,free_after=shutil.disk_usage('/dev/shm').free,
 records=records,rsna_touched=False,system_cache_dropped=False,gpu_reset=False)
print(json.dumps(result))
'''
    response = subprocess.run([
        'ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'HostKeyAlias=13.220.240.128', 'ubuntu@3.226.249.134',
        '/home/ubuntu/venv/bin/python -',
    ], input=code, capture_output=True, text=True, check=True, timeout=60)
    result = json.loads(response.stdout)
    if args.execute:
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
