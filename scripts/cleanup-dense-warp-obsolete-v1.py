"""Remove two exact backed-up Biohub RAM targets, preserving the live movie run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    receipt = ROOT / 'reports/experiments/dense-warp-obsolete-v1-cleanup.json'
    if receipt.exists():
        raise ValueError('Preserve completed cleanup receipt')
    backup = json.loads((ROOT / 'reports/experiments/dense-warp-v1-smoke-harvest.json').read_text())
    assert backup['status'] == 'verified_backup'
    records = []
    for row in backup['records']:
        local = ROOT / '.biohub/cache/dense-warp-v1-smoke-output' / row['path']
        assert local.stat().st_size == row['bytes']
        assert hashlib.sha256(local.read_bytes()).hexdigest() == row['sha256']
        records.append(dict(row, path='/dev/shm/biohub-dense-warp-v1-smoke/' + row['path'], recovery=str(local)))
    local = ROOT / '.biohub/cache/dense-warp-movie-v1-bundle.tar'
    digest = 'be8bf51d17f73b1e12400554882bf6055e54d6cea1c932dc3821e6ab0a24606c'
    assert local.stat().st_size == 82411520 and hashlib.sha256(local.read_bytes()).hexdigest() == digest
    records.append(dict(path='/dev/shm/biohub-dense-warp-movie-v1.tar', bytes=82411520, sha256=digest, recovery=str(local)))
    code = '''import datetime,hashlib,json,os,shutil
from pathlib import Path
''' + f'records={records!r}\nexecute={args.execute!r}\n' + '''
root=Path('/dev/shm/biohub-dense-warp-v1-smoke')
archive=Path('/dev/shm/biohub-dense-warp-movie-v1.tar')
assert root.resolve(strict=True)==root and not root.is_symlink()
assert json.loads((root/'RESULT.json').read_text())['status']=='dense_warp_smoke_passed'
expected={r['path'] for r in records}
assert len(expected)==4
assert {str(p) for p in root.iterdir()}==expected-{str(archive)}
for row in records:
 p=Path(row['path'])
 assert p.parent==root or p==archive
 assert p.resolve(strict=True)==p and not p.is_symlink() and p.is_file()
 assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  args=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\\0')]
  assert not any(a in expected or a==str(root) or a.startswith(str(root)+'/') for a in args), 'Target referenced by process'
  links=[proc/'cwd',*(proc/'fd').iterdir()]
  for link in links:
   try:target=os.readlink(link)
   except (FileNotFoundError,PermissionError):continue
   assert target not in expected and target!=str(root) and not target.startswith(str(root)+'/'), 'Target in use'
 except (FileNotFoundError,PermissionError):continue
before=shutil.disk_usage('/dev/shm').free
if execute:
 for row in records:Path(row['path']).unlink()
 root.rmdir()
 assert all(not Path(r['path']).exists() for r in records)
print(json.dumps(dict(status='removed_verified_ram_copies' if execute else 'verified_dry_run',
 timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 bytes=sum(r['bytes'] for r in records),records=records,
 free_before=before,free_after=shutil.disk_usage('/dev/shm').free,
 rsna_touched=False,gpu_reset=False,system_cache_dropped=False,live_movie_run_touched=False)))
'''
    result = subprocess.run(['ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=13.220.240.128',
        'ubuntu@3.226.249.134', '/home/ubuntu/venv/bin/python -'], input=code, capture_output=True,
        text=True, check=True, timeout=60)
    data = json.loads(result.stdout)
    if args.execute:
        receipt.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in data.items() if k != 'records'}))


if __name__ == '__main__':
    main()
