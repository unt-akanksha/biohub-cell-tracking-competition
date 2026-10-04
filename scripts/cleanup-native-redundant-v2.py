"""Remove only four exact backed-up terminal checkpoint copies on Antelume."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    receipt=ROOT/'reports/experiments/native-terminal-checkpoint-cleanup-v2.json'
    if receipt.exists():raise ValueError('Preserve completed cleanup receipt')
    selected={
        'native-correspondence-v2-training-smoke-r3':{'resume.pt','6bba-resnet3d/best.pt','6bba-token_transformer3d/best.pt'},
        'native-correspondence-v2-training-full':{'resume.pt'}}
    records=[]
    for name,names in selected.items():
        backup=json.loads((ROOT/'reports/experiments'/(name+'-harvest.json')).read_text())
        if backup['status']!='verified_backup':raise ValueError('Verified recovery required')
        for record in backup['records']:
            if record['path'] not in names:continue
            local=(ROOT/'.biohub/cache'/(name+'-output')/record['path']).resolve(strict=True)
            if sha(local)!=record['sha256'] or local.stat().st_size!=record['bytes']:raise ValueError('Local recovery changed')
            records.append(dict(relative=name+'/'+record['path'],sha256=record['sha256'],bytes=record['bytes'],local_recovery=str(local)))
    if len(records)!=4:raise ValueError('Expected four exact copies')
    code=f'''import hashlib,json,subprocess,shutil
from pathlib import Path
root=Path('/tmp/biohub-image-context-v2.ScdSdY').resolve(strict=True)
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
assert json.loads((root/'native-correspondence-v2-training-full/RESULT.json').read_text())['status']=='training_complete'
records={records!r};targets=[]
for row in records:
 raw=root/row['relative'];assert not raw.is_symlink()
 path=raw.resolve(strict=True);assert path.is_relative_to(root) and path.is_file() and path.name in ('best.pt','resume.pt')
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 targets.append(path)
before=shutil.disk_usage(root).free
for path in targets:path.unlink()
print(json.dumps(dict(status='removed_verified_redundant_terminal_copies',records=records,
    recovered_bytes=sum(r['bytes'] for r in records),free_before=before,free_after=shutil.disk_usage(root).free,
    all_four_full_training_best_weights_retained=True,rsna_touched=False)))
'''
    command=['ssh','-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
             '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128','ubuntu@3.226.249.134',
             '/home/ubuntu/venv/bin/python -c '+shlex.quote(code)]
    response=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
    result=json.loads(response.stdout);receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
