"""Recover each terminal complete-pipeline run without modifying remote outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
SSH=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']
HOST='ubuntu@3.226.249.134'


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','full'),required=True);args=p.parse_args()
    name='dense-warp-movie-v1-'+args.mode;remote='/dev/shm/biohub-'+name
    local=ROOT/'.biohub/cache'/(name+'-output');assert not local.exists()
    code='''from pathlib import Path
import hashlib,json
root=Path('''+repr(remote)+''')
result=json.loads((root/'result.json').read_text())
assert result['status'] in ('functionality_passed','complete_prelabel_predictions','failed')
records=[]
for p in sorted(root.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():records.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(records))
'''
    r=subprocess.run(['ssh',*SSH,HOST,'/home/ubuntu/venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=60)
    records=json.loads(r.stdout)
    subprocess.run(['scp',*SSH,'-r',HOST+':'+remote,str(local)],check=True,timeout=180)
    assert {p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file()}=={r['path'] for r in records}
    for row in records:
        path=local/row['path'];assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    shutil.copy2(local/'result.json',ROOT/'reports/experiments'/(name+'-result.json'))
    receipt=dict(status='verified_backup',bytes=sum(r['bytes'] for r in records),records=records,remote_files_deleted=False)
    (ROOT/'reports/experiments'/(name+'-harvest.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'}))


if __name__=='__main__':main()
