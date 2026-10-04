"""Incrementally back up only completed, hash-verified owned GPU predictions.

No labels, remote writes, deletion, GPU calls or process interruption. Failed
partial transfers remain in uniquely named folders for inspection/recovery.
"""
import argparse
import ipaddress
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.public_d4_full_movie import sha,STEMS

REMOTE='/tmp/biohub-d4-preflight-v1.o4fsjO/full-v1'
KEY='C:/Users/IndarKumar/.ssh/rsna_ec2'


def command(args):
    subprocess.run(args,check=True,timeout=240)


def main(args):
    ipaddress.ip_address(args.host)
    host='ubuntu@'+args.host
    options=['-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=15']
    output=ROOT/'.biohub/cache/public-d4-full-movie-v1-output'
    output.mkdir(parents=True,exist_ok=True)
    terminal=subprocess.run(['ssh',*options,host,'test -f '+REMOTE+'/result.json'],
                            timeout=30).returncode
    if terminal not in (0,1):raise RuntimeError('Cannot determine owned run terminal')
    source='result.json' if terminal==0 else 'prelabel-progress.json'
    snapshot=output/('snapshot-'+uuid.uuid4().hex+'.json')
    command(['scp','-q',*options,host+':'+REMOTE+'/'+source,str(snapshot)])
    receipt=json.loads(snapshot.read_text())
    contract=ROOT/'.biohub/cache/public-d4-full-movie-v1-bundle/CONTRACT.json'
    if (receipt.get('run_id')!='public-d4-full-movie-v1' or receipt.get('mode')!='full'
            or receipt.get('contract_sha256')!=sha(contract)
            or receipt.get('authorized_for_submission') is not False):
        raise ValueError('Unexpected transport receipt')
    copied=[]
    for stem,arms in receipt['movies'].items():
        if stem not in STEMS or not set(arms).issubset({'original','corrected'}):
            raise ValueError('Unknown completed movie/arm')
        for arm,row in arms.items():
            if row.get('frames')!=100:raise ValueError('Not a complete movie')
            name=stem+'-'+arm; destination=output/name
            if destination.exists():
                if sha(destination/'prediction.json')!=row['prediction_sha256']:
                    raise ValueError('Existing local completed graph changed')
                continue
            temporary=output/('transfer-'+uuid.uuid4().hex);temporary.mkdir()
            command(['scp','-q','-r',*options,host+':'+REMOTE+'/'+name,str(temporary)])
            fetched=temporary/name
            if sha(fetched/'prediction.json')!=row['prediction_sha256']:
                raise ValueError('Transferred prediction checksum failed')
            fetched.rename(destination)
            copied.append(name)
    if source=='result.json':
        final=output/'result.json'
        if final.exists() and sha(final)!=sha(snapshot):
            raise ValueError('Refuse to replace a previous terminal receipt')
        if not final.exists():shutil.copyfile(snapshot,final)
    print(json.dumps(dict(status=receipt['status'],snapshot_sha256=sha(snapshot),
        newly_backed_up=copied,total_completed=sum(len(a) for a in receipt['movies'].values()),
        terminal=source=='result.json',ground_truth_opened=False,output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True)
    main(p.parse_args())
