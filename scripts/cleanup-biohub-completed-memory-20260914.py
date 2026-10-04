"""Remove two completed, exactly backed-up Biohub RAM directories only."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ('/dev/shm/biohub-dense-warp-movie-v1-full',
     '.biohub/cache/dense-warp-movie-v1-full-output',
     'reports/experiments/dense-warp-movie-v1-full-harvest.json'),
    ('/dev/shm/biohub-dense-warp-v1-full',
     '.biohub/cache/dense-warp-v1-full-output',
     'reports/experiments/dense-warp-v1-full-harvest.json'),
]


def main():
    payload = []
    for remote, local, receipt in TARGETS:
        backup = ROOT / local
        manifest = json.loads((ROOT / receipt).read_text())
        assert manifest['status'] == 'verified_backup'
        records = manifest['records']
        assert {p.relative_to(backup).as_posix() for p in backup.rglob('*') if p.is_file()} == {r['path'] for r in records}
        for record in records:
            path = backup / record['path']
            assert not path.is_symlink()
            assert path.stat().st_size == record['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        payload.append(dict(remote=remote, local=str(backup), records=records))
    code = '''import hashlib,json,os,shutil,time
from pathlib import Path
payload = ''' + repr(payload) + '''
targets = [Path(item['remote']) for item in payload]
for root in targets:
 assert root.parent == Path('/dev/shm') and root.name in ('biohub-dense-warp-movie-v1-full','biohub-dense-warp-v1-full')
 assert root.is_dir() and not root.is_symlink() and root.resolve() == root
# Check command lines, mapped files, working directories and open descriptors.
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
 try:
  raw = (proc/'cmdline').read_bytes() + (proc/'maps').read_bytes()
  paths = [proc/'cwd', *list((proc/'fd').iterdir())]
  for link in paths:
   try: raw += os.readlink(link).encode()
   except FileNotFoundError: pass
  assert not any(str(root).encode() in raw for root in targets), 'Target still referenced by PID '+proc.name
 except (FileNotFoundError,ProcessLookupError): pass
 except PermissionError:
  raise RuntimeError('Cannot safely inspect process '+proc.name)
for item,root in zip(payload,targets):
 files = sorted(root.rglob('*'))
 assert not any(p.is_symlink() for p in files)
 assert {p.relative_to(root).as_posix() for p in files if p.is_file()} == {r['path'] for r in item['records']}
 for r in item['records']:
  path = root/r['path']
  assert path.stat().st_size == r['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256']
removed=[]
for item,root in zip(payload,targets):
 shutil.rmtree(root)
 assert not root.exists()
 removed.append(dict(remote=str(root),backup=item['local'],bytes=sum(r['bytes'] for r in item['records']),files=len(item['records'])))
print(json.dumps(dict(status='completed_backed_up_ram_cleanup',removed=removed,bytes=sum(x['bytes'] for x in removed),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),active_run_untouched=True,foreign_jobs_untouched=True)))
'''
    result = subprocess.run([
        'ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes', '-o',
        'HostKeyAlias=13.220.240.128', 'ubuntu@3.226.249.134',
        'sudo -n /home/ubuntu/venv/bin/python -'], input=code, text=True,
        capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr)
    report = json.loads(result.stdout)
    (ROOT/'reports/experiments/biohub-completed-memory-cleanup-20260914.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
