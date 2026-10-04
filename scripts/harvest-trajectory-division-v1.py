"""Read-only, hash-verified harvest of this exact owned smoke/full run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SSH = ['-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'HostKeyAlias=13.220.240.128']
HOST = 'ubuntu@3.226.249.134'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024 ** 2), b''):
            h.update(part)
    return h.hexdigest()


def main(mode, experiment='division'):
    if experiment not in ('division', 'amp-encode', 'overlap', 'overlap-cache'):
        raise ValueError('Only explicitly owned trajectory experiments may be harvested')
    remote = f'/tmp/biohub-image-context-v2.ScdSdY/trajectory-{experiment}-{mode}-v1'
    code = f"""from pathlib import Path
import hashlib,json
root=Path({remote!r}).resolve(strict=True)
terminal=json.loads((root/'result.json').read_text())
if terminal['mode']!={mode!r}:raise ValueError('Wrong terminal mode')
if terminal['status'] not in ('functionality_passed','complete_prelabel_predictions','failed'):raise ValueError('Not terminal')
records=[]
for p in sorted(root.rglob('*')):
 if not p.is_file():continue
 if not p.resolve().is_relative_to(root) or p.is_symlink():raise ValueError('Escaping artifact')
 records.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(status=terminal['status'],files=records)))
"""
    response = subprocess.run(['ssh', *SSH, HOST, '/home/ubuntu/venv/bin/python', '-'],
                              input=code, capture_output=True, text=True, timeout=120, check=True)
    inventory = json.loads(response.stdout)
    output = ROOT / f'.biohub/cache/trajectory-{experiment}-{mode}-v1-output'
    output.mkdir(parents=True, exist_ok=True)
    def copy_record(row):
        name = PurePosixPath(row['path'])
        if name.is_absolute() or '..' in name.parts or any(' ' in part for part in name.parts):
            raise ValueError('Unsafe artifact path')
        if row['bytes'] > 100 * 1024 ** 2:
            raise ValueError('Unexpectedly large inference artifact')
        path = output / row['path']; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run(['scp', *SSH, HOST + ':' + remote + '/' + row['path'], str(path)], check=True, timeout=180)
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Downloaded artifact mismatch; preserve for investigation')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy_record, inventory['files']))
    path = output / 'REMOTE_ARTIFACT_MANIFEST.json'
    if path.exists() and json.loads(path.read_text()) != inventory:
        raise ValueError('Preserve old inventory')
    if not path.exists():
        path.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(status='hash_verified_local', mode=mode, experiment=experiment, files=len(inventory['files']),
                         bytes=sum(r['bytes'] for r in inventory['files']), manifest_sha256=sha(path))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('smoke', 'full'), required=True)
    p.add_argument('--experiment', choices=('division', 'amp-encode', 'overlap', 'overlap-cache'), default='division')
    args = p.parse_args()
    main(args.mode, args.experiment)
