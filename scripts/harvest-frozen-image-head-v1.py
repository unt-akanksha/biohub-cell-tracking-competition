"""Read-only verified backup of the two completed frozen-head stages."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/tmp/biohub-image-context-v2.ScdSdY'
HOST='ubuntu@3.226.249.134'
OPTIONS=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
         '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main(family='frozen-head-v1'):
    choices={'frozen-head-v1':('frozen-head-v1-smoke','frozen-head-v1-full','frozen-image-head-v1-output'),
             'zebrahub-transfer-v1':('zebrahub-transfer-v1-smoke','zebrahub-transfer-v1-full','zebrahub-division-transfer-v1-output')}
    smoke,full,local_name=choices[family]
    code='''from pathlib import Path
import hashlib,json
root=Path('/tmp/biohub-image-context-v2.ScdSdY')
files=[]
for name in STAGE_NAMES:
    folder=root/name
    if not (folder/'result.json').is_file():raise RuntimeError('Stage not terminal')
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        if path.suffix=='.partial':raise RuntimeError('Unexpected partial file')
        if root.resolve() not in path.resolve().parents:raise ValueError('Path escaped owned root')
        files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(dict(files=files)))
'''
    code=code.replace('STAGE_NAMES',repr((smoke,full)))
    query=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/venv/bin/python','-'],input=code,
                         text=True,capture_output=True,check=True,timeout=60)
    inventory=json.loads(query.stdout);destination=ROOT/'.biohub/cache'/local_name
    destination.mkdir(exist_ok=True)
    for row in inventory['files']:
        name=Path(row['path'])
        if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe artifact name')
        path=destination/name
        if path.exists():
            if sha(path)!=row['sha256']:raise ValueError('Immutable local artifact differs')
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.transfer-'+uuid.uuid4().hex)
        subprocess.run(['scp','-q',*OPTIONS,HOST+':'+REMOTE+'/'+row['path'],str(temporary)],check=True,timeout=300)
        if temporary.stat().st_size!=row['bytes'] or sha(temporary)!=row['sha256']:
            raise ValueError('Transport verification failed')
        temporary.rename(path)
        print(json.dumps(dict(event='artifact_verified',**row)),flush=True)
    receipt=destination/('inventory-'+uuid.uuid4().hex+'.json')
    receipt.write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(status='all_artifacts_verified',files=len(inventory['files']))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=('frozen-head-v1','zebrahub-transfer-v1'),default='frozen-head-v1')
    main(parser.parse_args().family)
