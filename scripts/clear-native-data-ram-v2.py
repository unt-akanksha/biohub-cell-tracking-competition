"""Remove only a fully verified, recoverable Biohub /dev/shm dataset duplicate."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/dev/shm/biohub-native-correspondence-v2-full'
EXPECTED = 'f2861ce9f21307507ed521716ca3bf9a7adf3b76ce9f8216e4407233616c693a'
PROGRESS = '9f7ed5e16647284d937c305a61133a6b8e733eb962ac4760ee1331ed7271f076'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = ROOT / 'reports/experiments/native-correspondence-v2-ram-cleanup.json'
    if receipt.exists():
        raise ValueError('Preserve completed cleanup receipt')
    local = ROOT / '.biohub/cache/native-correspondence-v2-data'
    manifest_path = local / 'RESULT.json'
    if sha(manifest_path) != EXPECTED:
        raise ValueError('Local recovery manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if sha(local / 'PROGRESS.json') != PROGRESS:
        raise ValueError('Intermediate progress log must also be recoverable')
    for record in manifest['records']:
        path = local / record['path']
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Local recovery packet changed')
    code = '''import hashlib,json,subprocess,shutil
from pathlib import Path
raw=Path(''' + repr(REMOTE) + ''')
assert not raw.is_symlink()
root=raw.resolve(strict=True)
assert str(root)==''' + repr(REMOTE) + ''' and root.parent==Path('/dev/shm')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
manifest_path=root/'RESULT.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==''' + repr(EXPECTED) + '''
manifest=json.loads(manifest_path.read_text())
assert manifest['status']=='data_complete'
assert hashlib.sha256((root/'PROGRESS.json').read_bytes()).hexdigest()==''' + repr(PROGRESS) + '''
expected={'RESULT.json','PROGRESS.json'}|{r['path'] for r in manifest['records']}
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual==expected, (len(actual),len(expected),sorted(actual-expected))
assert all(not p.is_symlink() for p in root.rglob('*'))
for r in manifest['records']:
 p=(root/r['path']).resolve(strict=True)
 assert p.is_relative_to(root) and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
before=shutil.disk_usage('/dev/shm').free
total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
shutil.rmtree(root)
print(json.dumps(dict(status='verified_biohub_ram_duplicate_removed',remote_root=str(root),bytes=total,free_before=before,free_after=shutil.disk_usage('/dev/shm').free,rsna_touched=False,gpu_reset=False)))
'''
    command = ['ssh', '-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=13.220.240.128', 'ubuntu@3.226.249.134',
               '/home/ubuntu/venv/bin/python -c ' + shlex.quote(code)]
    response = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if response.returncode:
        raise RuntimeError('Remote cleanup guard refused: ' + response.stderr[-2000:])
    result = json.loads(response.stdout)
    result.update(local_recovery=str(local), manifest_sha256=EXPECTED, local_packets_verified=len(manifest['records']))
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
