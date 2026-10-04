"""Reclaim exact completed source image cache, recoverable from Kaggle archive."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SSH=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']


def main():
    # All nonrecoverable experiment outputs must have exact local backups first.
    cache=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-full-output'
    receipt=json.loads((ROOT/'reports/experiments/trajectory-disagreement-source-v1-full-harvest.json').read_text())
    assert receipt['status']=='verified_backup'
    for r in receipt['records']:
        assert hashlib.sha256((cache/r['path']).read_bytes()).hexdigest()==r['sha256']
    manifest=ROOT/'.biohub/cache/trajectory-disagreement-source-v1-plan/IMAGE_MANIFEST.json'
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest=='8d7bb65e42d87488ea6b8023b78118ab45e4b43b3fc21cb6122e7a7f72f4c779'
    # Verify a fresh authorized archive plan exists before retiring recoverable images.
    new_plan=ROOT/'.biohub/cache/trajectory-ranker-selection-v1-plan/PRIVATE_ARCHIVE_PLAN.json'
    new=json.loads(new_plan.read_text())
    assert len(new['records'])==1020 and new['movie_plan_sha256']=='1d9cbb244bbdc23d631c45811ccd222bb2e29d5b70a2cd6f05fa87aeb5ff297e'
    code='''import hashlib,json,os,shutil,time
from pathlib import Path
root=Path('/dev/shm/biohub-disagreement-source-v1.nRsU8k/images')
assert root.resolve()==root and root.parent==Path('/dev/shm/biohub-disagreement-source-v1.nRsU8k') and not root.is_symlink()
terminal=json.loads((root.parent/'full/result.json').read_text())
assert terminal['status']=='complete_prelabel_predictions' and len(terminal['movies'])==8
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  raw=(proc/'cmdline').read_bytes()+(proc/'maps').read_bytes()
  for p in [proc/'cwd',*list((proc/'fd').iterdir())]:
   try:raw+=os.readlink(p).encode()
   except FileNotFoundError:pass
  assert str(root).encode() not in raw,'Live image reference PID '+proc.name
 except (FileNotFoundError,ProcessLookupError):pass
manifest=root/'IMAGE_MANIFEST.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='''+repr(digest)+'''
data=json.loads(manifest.read_text())
assert data['status']=='complete' and len(data['records'])==816
files=list(root.rglob('*'))
assert not any(p.is_symlink() for p in files)
assert {p.relative_to(root).as_posix() for p in files if p.is_file()}=={r['path'] for r in data['records']}|{'IMAGE_MANIFEST.json'}
for r in data['records']:
 p=root/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
removed=sum(p.stat().st_size for p in files if p.is_file())
shutil.rmtree(root)
assert not root.exists()
print(json.dumps(dict(status='retired_completed_recoverable_image_cache',path=str(root),bytes=removed,raw_images_locally_backed_up=False,recovery='Kaggle competition archive, saved member CRC and SHA256 manifest',experiment_outputs_preserved=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
'''
    result=subprocess.run(['ssh',*SSH,'ubuntu@3.226.249.134','sudo -n /home/ubuntu/venv/bin/python -'],input=code,text=True,capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr)
    report=json.loads(result.stdout)
    (ROOT/'reports/experiments/trajectory-source-images-v1-cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
