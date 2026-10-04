"""Back up exact terminal native-data artifacts; delete nothing on Antelume."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
SSH=['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']
HOST='ubuntu@3.226.249.134'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
    return digest.hexdigest()


def main():
    start=time.monotonic(); remote='/dev/shm/biohub-native-correspondence-v2-full'
    completed=subprocess.run(['ssh',*SSH,HOST,'cat',remote+'/RESULT.json'],capture_output=True,check=True,timeout=30)
    state=json.loads(completed.stdout)
    if state['status']!='data_complete' or len(state['completed_movies'])!=101:
        raise ValueError('Require complete 101-movie native dataset')
    output=ROOT/'.biohub/cache/native-correspondence-v2-data'; output.mkdir(exist_ok=False)
    subprocess.run(['scp',*SSH,'-r',HOST+':'+remote+'/data',str(output)],check=True,timeout=1800)
    subprocess.run(['scp',*SSH,HOST+':'+remote+'/RESULT.json',str(output/'RESULT.json')],check=True,timeout=60)
    if json.loads((output/'RESULT.json').read_text())!=state:
        raise ValueError('Terminal manifest changed during backup')
    total=0
    for record in state['records']:
        path=(output/record['path']).resolve()
        if not path.is_relative_to(output.resolve()) or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Local data backup failed verification')
        total+=record['bytes']
    receipt=dict(status='verified_local_backup',packets=len(state['records']),bytes=total,
                 result_sha256=sha(output/'RESULT.json'),local_root=str(output),remote_root=remote,
                 remote_files_deleted=False,seconds=time.monotonic()-start)
    (ROOT/'reports/experiments/native-correspondence-v2-data-harvest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__': main()
