"""Retire only completed, archive-recoverable Biohub selection images from RAM."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    backup = ROOT / '.biohub/cache/trajectory-ranker-selection-v1-full-output'
    receipt = json.loads((ROOT / 'reports/experiments/trajectory-ranker-selection-v1-full-harvest.json').read_text())
    assert receipt['status'] == 'verified_backup'
    assert {p.relative_to(backup).as_posix() for p in backup.rglob('*') if p.is_file()} == {r['path'] for r in receipt['records']}
    for record in receipt['records']:
        path = backup / record['path']
        assert not path.is_symlink()
        assert path.stat().st_size == record['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    plan = ROOT / '.biohub/cache/trajectory-ranker-selection-v1-plan'
    digest = hashlib.sha256((plan / 'IMAGE_MANIFEST.json').read_bytes()).hexdigest()
    assert digest == '18a13bbe5c582fd58cbc48687339a1ee33cef88a94ecd2299f99f5c6310fce2f'
    archive = json.loads((plan / 'PRIVATE_ARCHIVE_PLAN.json').read_text())
    assert len(archive['records']) == 1020
    assert archive['movie_plan_sha256'] == '1d9cbb244bbdc23d631c45811ccd222bb2e29d5b70a2cd6f05fa87aeb5ff297e'
    code = '''import hashlib,json,os,shutil,time
from pathlib import Path
root = Path('/dev/shm/biohub-ranker-selection-v1.LJ0JSl/images')
assert root.is_dir() and not root.is_symlink() and root.resolve() == root
assert root.parent == Path('/dev/shm/biohub-ranker-selection-v1.LJ0JSl')
terminal = json.loads((root.parent/'full/result.json').read_text())
assert terminal['status'] == 'complete_prelabel_predictions' and len(terminal['movies']) == 10
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
 try:
  raw = (proc/'cmdline').read_bytes() + (proc/'maps').read_bytes()
  for link in [proc/'cwd', *list((proc/'fd').iterdir())]:
   try: raw += os.readlink(link).encode()
   except FileNotFoundError: pass
  assert str(root).encode() not in raw, 'Live reference PID ' + proc.name
 except (FileNotFoundError, ProcessLookupError): pass
manifest = root/'IMAGE_MANIFEST.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == ''' + repr(digest) + '''
data = json.loads(manifest.read_text())
assert data['status'] == 'complete' and len(data['records']) == 1020
files = list(root.rglob('*'))
assert not any(p.is_symlink() for p in files)
assert {p.relative_to(root).as_posix() for p in files if p.is_file()} == {r['path'] for r in data['records']} | {'IMAGE_MANIFEST.json'}
for record in data['records']:
 path = root/record['path']
 assert path.stat().st_size == record['bytes']
 assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
removed = sum(p.stat().st_size for p in files if p.is_file())
shutil.rmtree(root)
assert not root.exists()
print(json.dumps(dict(status='retired_completed_recoverable_image_cache',path=str(root),bytes=removed,raw_images_locally_backed_up=False,recovery='Kaggle competition archive with saved member CRC and SHA256 manifest',experiment_outputs_preserved=True,foreign_jobs_untouched=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
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
    (ROOT / 'reports/experiments/trajectory-selection-images-v1-cleanup.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
