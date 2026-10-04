"""Recover exact owned terminal experiment artifacts and verify every byte."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

ROOT=Path(__file__).resolve().parents[1]
SSH=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']
HOST='ubuntu@3.226.249.134'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');args=parser.parse_args()
    name='dense-warp-v1-'+('full' if args.full else 'smoke')
    remote='/dev/shm/biohub-'+name;local=ROOT/'.biohub/cache'/(name+'-output')
    assert not local.exists()
    code='''from pathlib import Path
import json,hashlib
root=Path('''+repr(remote)+''')
assert json.loads((root/'RESULT.json').read_text())['status'] in ('dense_warp_smoke_passed','dense_warp_training_complete','failed')
records=[]
for p in sorted(root.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():records.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(records))
'''
    response=subprocess.run(['ssh',*SSH,HOST,'/home/ubuntu/venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=60)
    records=json.loads(response.stdout)
    subprocess.run(['scp',*SSH,'-r',HOST+':'+remote,str(local)],check=True,timeout=180)
    assert {p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file()}=={r['path'] for r in records}
    for r in records:
        p=local/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    shutil.copy2(local/'RESULT.json',ROOT/'reports/experiments'/(name+'-result.json'))
    result=dict(status='verified_backup',bytes=sum(r['bytes'] for r in records),records=records,remote_files_deleted=False)
    (ROOT/'reports/experiments'/(name+'-harvest.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
