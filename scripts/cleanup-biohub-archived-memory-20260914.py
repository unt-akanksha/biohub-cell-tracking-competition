"""Clear three exact archived RAM outputs, never active runs or global caches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    ('/dev/shm/biohub-disagreement-source-v1.nRsU8k/full', 'trajectory-disagreement-source-v1-full'),
    ('/dev/shm/biohub-ranker-selection-v1.LJ0JSl/full', 'trajectory-ranker-selection-v1-full'),
    ('/dev/shm/biohub-dense-warp-movie-v1-smoke', 'dense-warp-movie-v1-smoke'),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    receipt = ROOT / 'reports/experiments/biohub-archived-memory-cleanup-20260914.json'
    if receipt.exists():
        raise ValueError('Preserve completed cleanup receipt')
    payload = []
    for remote, name in TARGETS:
        backup = ROOT / '.biohub/cache' / (name + '-output')
        manifest = json.loads((ROOT / 'reports/experiments' / (name + '-harvest.json')).read_text())
        assert manifest['status'] == 'verified_backup'
        records = manifest['records']
        assert {p.relative_to(backup).as_posix() for p in backup.rglob('*') if p.is_file()} == {r['path'] for r in records}
        for row in records:
            path = backup / row['path']
            assert path.resolve().is_relative_to(backup.resolve()) and not path.is_symlink()
            assert path.stat().st_size == row['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        payload.append(dict(remote=remote, backup=str(backup), records=records))
    code = '''import datetime,hashlib,json,os,shutil
from pathlib import Path
''' + f'payload={payload!r}\nexecute={args.execute!r}\nallowed={tuple(r for r, _ in TARGETS)!r}\n' + '''
roots=[Path(row['remote']) for row in payload]
for root in roots:
 assert str(root) in allowed and root.is_dir() and not root.is_symlink()
 assert root.resolve(strict=True)==root and root.is_relative_to(Path('/dev/shm'))
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  raw=(proc/'cmdline').read_bytes()+(proc/'maps').read_bytes()
  for link in [proc/'cwd',*list((proc/'fd').iterdir())]:
   try:raw+=os.readlink(link).encode()
   except FileNotFoundError:pass
  assert not any(str(root).encode() in raw for root in roots), 'Target referenced by PID '+proc.name
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError:raise RuntimeError('Cannot safely inspect PID '+proc.name)
for item,root in zip(payload,roots):
 files=list(root.rglob('*'))
 assert not any(p.is_symlink() for p in files)
 assert {p.relative_to(root).as_posix() for p in files if p.is_file()}=={r['path'] for r in item['records']}
 for row in item['records']:
  path=root/row['path']
  assert path.resolve(strict=True).is_relative_to(root)
  assert path.stat().st_size==row['bytes']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
before=shutil.disk_usage('/dev/shm').free
if execute:
 for root in roots:
  shutil.rmtree(root)
  assert not root.exists()
summary=[dict(remote=row['remote'],backup=row['backup'],bytes=sum(r['bytes'] for r in row['records']),files=len(row['records'])) for row in payload]
print(json.dumps(dict(status='removed_verified_ram_outputs' if execute else 'verified_dry_run',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),targets=summary,
 bytes=sum(row['bytes'] for row in summary),free_before=before,free_after=shutil.disk_usage('/dev/shm').free,
 active_run_untouched=True,foreign_jobs_untouched=True,gpu_reset=False,system_cache_dropped=False)))
'''
    result = subprocess.run([
        'ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', '-o',
        'HostKeyAlias=13.220.240.128', 'ubuntu@3.226.249.134',
        'sudo -n /home/ubuntu/venv/bin/python -'], input=code, text=True,
        capture_output=True, check=True, timeout=60)
    report = json.loads(result.stdout)
    if args.execute:
        receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
