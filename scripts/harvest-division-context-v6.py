"""Recover the terminal context assets by one archive, including partial SCP repair."""
import hashlib
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[1]
SSH=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']
HOST='ubuntu@3.226.249.134'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training',choices=('r1','r2'));args=parser.parse_args()
    names=['native-division-context-v6-smoke','native-division-context-v6-full','native-division-context-v6-model-smoke']
    recovery_name='native-division-context-v6-recovery-v1'
    receipt_name='native-division-context-v6-harvest'
    if args.training:
        names=['native-division-context-v6-training-full'+('-r2' if args.training=='r2' else '')]
        if args.training=='r1':names+=['native-division-context-v6-numeric-diagnostic']
        recovery_name='native-division-context-v6-training-recovery-'+args.training
        receipt_name='native-division-context-v6-training-harvest-'+args.training
    code='''from pathlib import Path
import json,hashlib,tarfile
names='''+repr(names)+'''
archive=Path('''+repr('/dev/shm/biohub-'+recovery_name+'.tar')+''')
assert not archive.exists()
records=[]
with tarfile.open(archive,'w') as tf:
 for name in names:
  root=Path('/dev/shm/biohub-'+name).resolve(strict=True)
  assert json.loads((root/'RESULT.json').read_text())['status'] in ('context_smoke_passed','context_data_complete','temporal_model_smoke_passed','failed','temporal_training_complete')
  for p in sorted(root.rglob('*')):
   assert not p.is_symlink()
   if p.is_file():
    relative=p.relative_to(root).as_posix();records.append(dict(group=name,path=relative,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    tf.add(p,arcname=name+'/'+relative)
print(json.dumps(dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),records=records)))
'''
    response=subprocess.run(['ssh',*SSH,HOST,'/home/ubuntu/venv/bin/python -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=60)
    manifest=json.loads(response.stdout)
    archive=ROOT/'.biohub/cache'/(recovery_name+'.tar')
    if archive.exists():raise ValueError('Do not overwrite an existing recovery archive')
    subprocess.run(['scp',*SSH,HOST+':'+manifest['archive'],str(archive)],check=True,timeout=180)
    if sha(archive)!=manifest['archive_sha256']:raise ValueError('Recovery archive mismatch')
    expected={r['group']+'/'+r['path']:r for r in manifest['records']};repaired=0;copied=0
    for name in names:
        root=ROOT/'.biohub/cache'/(name+'-output')
        if root.exists():
            actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            allowed={r['path'] for r in manifest['records'] if r['group']==name}
            if not actual<=allowed:raise ValueError('Unexpected files in partial backup; preserve them')
    with tarfile.open(archive) as tf:
        if set(tf.getnames())!=set(expected):raise ValueError('Recovery member set changed')
        for member in tf.getmembers():
            r=expected[member.name];root=(ROOT/'.biohub/cache'/(r['group']+'-output')).resolve()
            destination=(root/r['path']).resolve()
            if not member.isfile() or not destination.is_relative_to(root):raise ValueError('Unsafe recovery path')
            if any(p.is_symlink() for p in [destination,*destination.parents] if p.exists()):raise ValueError('Symlink in recovery destination')
            data=tf.extractfile(member).read()
            if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Recovered member mismatch')
            if destination.exists() and sha(destination)==r['sha256']:continue
            repaired+=int(destination.exists());destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data);copied+=1
    for r in manifest['records']:
        path=ROOT/'.biohub/cache'/(r['group']+'-output')/r['path']
        if sha(path)!=r['sha256']:raise ValueError('Final backup verification failed')
    for name in names:shutil.copy2(ROOT/'.biohub/cache'/(name+'-output')/'RESULT.json',ROOT/'reports/experiments'/(name+'-result.json'))
    result=dict(status='verified_backup',archive_sha256=manifest['archive_sha256'],records=manifest['records'],
                bytes=sum(r['bytes'] for r in manifest['records']),new_or_recovered_files=copied,replaced_incomplete_files=repaired,
                remote_files_deleted=False)
    (ROOT/'reports/experiments'/(receipt_name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
