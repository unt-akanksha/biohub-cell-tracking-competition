"""Back up exact terminal source-collection artifacts, never its signed URL."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SSH=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']
HOST='ubuntu@3.226.249.134'


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','full'),required=True);args=p.parse_args()
    name='trajectory-disagreement-source-v1-'+args.mode
    remote='/dev/shm/biohub-disagreement-source-v1.nRsU8k/'+args.mode
    local=ROOT/'.biohub/cache'/(name+'-output');assert not local.exists()
    code='''import hashlib,json
from pathlib import Path
root=Path('''+repr(remote)+''')
result=json.loads((root/'result.json').read_text())
assert result['status'] in ('functionality_passed','complete_prelabel_predictions','failed')
assert result['run_id']=='trajectory-disagreement-source-v1'
records=[]
for path in sorted(root.rglob('*')):
 assert not path.is_symlink()
 if path.is_file():records.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(records))
'''
    response=subprocess.run(['ssh',*SSH,HOST,'/home/ubuntu/venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=60)
    records=json.loads(response.stdout)
    subprocess.run(['scp',*SSH,'-r',HOST+':'+remote,str(local)],check=True,timeout=180)
    assert {p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file()}=={r['path'] for r in records}
    for r in records:
        path=local/r['path'];assert path.stat().st_size==r['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
    shutil.copy2(local/'result.json',ROOT/'reports/experiments'/(name+'-result.json'))
    receipt=dict(status='verified_backup',bytes=sum(r['bytes'] for r in records),records=records,remote_files_deleted=False)
    (ROOT/'reports/experiments'/(name+'-harvest.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'}))


if __name__=='__main__':main()
