"""Copy only immutable completed model stages; never modify remote experiments."""
import argparse
import ipaddress
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.public_d4_full_movie import sha

REMOTE = '/tmp/biohub-image-context-v2.ScdSdY/pilot-v1'
OPTIONS = ['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
           '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']


def main(host):
    ipaddress.ip_address(host)
    host = 'ubuntu@' + host
    code = '''from pathlib import Path
import hashlib,json
root=Path('/tmp/biohub-image-context-v2.ScdSdY/pilot-v1')
terminal=root/'result.json'
files=[]
if terminal.exists():
    paths=[p for p in root.rglob('*') if p.is_file() and p.suffix!='.partial']
else:
    paths=[]
    for source in ('44b6','6bba'):
        folder=root/('source-'+source)
        if (folder/'terminal.json').exists(): paths.extend(p for p in folder.iterdir() if p.is_file())
for p in sorted(paths):
    assert root.resolve() in p.resolve().parents
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(terminal=terminal.exists(),files=files)))
'''
    query = subprocess.run(['ssh',*OPTIONS,host,'/home/ubuntu/venv/bin/python','-'],
        input=code, text=True, capture_output=True, check=True, timeout=90)
    inventory = json.loads(query.stdout)
    output = ROOT / '.biohub/cache/image-context-pilot-v2-output'
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in inventory['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe returned artifact path')
        destination = output / relative
        if destination.exists():
            if sha(destination) != row['sha256']:
                raise ValueError('Previously copied immutable artifact changed')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.transfer-' + uuid.uuid4().hex)
        subprocess.run(['scp','-q',*OPTIONS,host+':'+REMOTE+'/'+row['path'],str(temporary)],
                       check=True,timeout=900)
        if temporary.stat().st_size != row['bytes'] or sha(temporary) != row['sha256']:
            raise ValueError('Transferred artifact checksum mismatch')
        temporary.rename(destination)
        copied.append(row['path'])
        print(json.dumps(dict(event='artifact_verified',**row)),flush=True)
    snapshot = output / ('inventory-'+uuid.uuid4().hex+'.json')
    snapshot.write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(terminal=inventory['terminal'],files=len(inventory['files']),copied=copied)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    main(parser.parse_args().host)
