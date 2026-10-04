"""Verified model backups and optional removal of redundant smoke checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/tmp/biohub-image-context-v2.ScdSdY'
HOST='ubuntu@3.226.249.134'
OPTIONS=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
         '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def remote(code):
    run=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/venv/bin/python -c '+shlex.quote(code)],
                       capture_output=True,text=True,check=True,timeout=60)
    return json.loads(run.stdout)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--name',required=True,choices=(
        'native-correspondence-v2-training-smoke','native-correspondence-v2-training-smoke-r2',
        'native-correspondence-v2-training-smoke-r3','native-correspondence-v2-training-full'))
    parser.add_argument('--remove-redundant-smoke-checkpoints',action='store_true'); args=parser.parse_args()
    if args.remove_redundant_smoke_checkpoints and 'smoke' not in args.name: raise ValueError('Smoke-only cleanup')
    begin=time.monotonic(); output=ROOT/'.biohub/cache'/(args.name+'-output')
    receipt=ROOT/'reports/experiments'/(args.name+'-harvest.json')
    if receipt.exists(): raise ValueError('Preserve completed receipt')
    code=f'''import hashlib,json
from pathlib import Path
root=Path({(REMOTE+'/'+args.name)!r}).resolve(strict=True)
result=json.loads((root/'RESULT.json').read_text())
assert result['status'] in ('smoke_passed','training_complete') and result['run_id']=='native-correspondence-v2'
names=['RESULT.json','resume.pt']
for member in result['members']:
 assert member['embryo'] in ('6bba','44b6') and member['family'] in ('resnet3d','token_transformer3d')
 names.extend(member['embryo']+'-'+member['family']+'/'+n for n in ('best.pt','HISTORY.json','RESULT.json'))
records=[]
for name in names:
 path=(root/name).resolve(strict=True); assert path.is_relative_to(root) and path.is_file() and path.stat().st_size<512*1024**2
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 records.append(dict(path=name,bytes=path.stat().st_size,sha256=digest))
print(json.dumps(dict(root=str(root),records=records)))
'''
    inventory=remote(code); output.mkdir(exist_ok=True)
    def fetch(record):
        path=(output/record['path']).resolve()
        if not path.is_relative_to(output.resolve()): raise ValueError('Escaped backup root')
        path.parent.mkdir(exist_ok=True,parents=True)
        if path.exists():
            if sha(path)!=record['sha256']: raise ValueError('Preserve differing backup')
        else:
            temporary=path.with_suffix(path.suffix+'.download')
            if temporary.exists(): raise ValueError('Existing incomplete backup')
            subprocess.run(['scp',*OPTIONS,HOST+':'+inventory['root']+'/'+record['path'],str(temporary)],
                           capture_output=True,check=True,timeout=300)
            if temporary.stat().st_size!=record['bytes'] or sha(temporary)!=record['sha256']: raise ValueError('Transfer verification failed')
            temporary.replace(path)
        if path.stat().st_size!=record['bytes']: raise ValueError('Local size mismatch')
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,inventory['records']))
    result=dict(status='verified_backup',local_root=str(output),remote_root=inventory['root'],records=inventory['records'],
                total_bytes=sum(r['bytes'] for r in inventory['records']),remote_files_deleted=False)
    if args.remove_redundant_smoke_checkpoints:
        chosen=[r for r in inventory['records'] if r['path']=='resume.pt' or r['path'].endswith('/best.pt')]
        for record in chosen:
            if sha(output/record['path'])!=record['sha256']:raise ValueError('Recheck failed')
        cleanup=f'''import hashlib,json,subprocess,shutil
from pathlib import Path
root=Path({inventory['root']!r}).resolve(strict=True)
assert root.parent==Path({REMOTE!r}).resolve(strict=True) and 'smoke' in root.name
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
records={chosen!r}; targets=[]
for r in records:
 raw=root/r['path']; assert not raw.is_symlink()
 path=raw.resolve(strict=True); assert path.is_relative_to(root) and path.is_file()
 assert path.stat().st_size==r['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
 targets.append(path)
before=shutil.disk_usage(root).free
for path in targets:path.unlink()
print(json.dumps(dict(removed=records,recovered_bytes=sum(r['bytes'] for r in records),free_before=before,free_after=shutil.disk_usage(root).free)))
'''
        result['cleanup']=remote(cleanup); result['remote_files_deleted']=True
    result['seconds']=time.monotonic()-begin
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','cleanup')}))


if __name__=='__main__':main()
