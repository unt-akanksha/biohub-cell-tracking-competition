"""Append new optimization positives; all original features/labels must be exact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old',type=Path,required=True);parser.add_argument('--extra',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    old_path=args.old/'RESULT.json';extra_path=args.extra/'RESULT.json'
    if sha(old_path)!='50173b81c2798f8524a14ff9b8d11752a7ea714694184f26e943846db5722130':raise ValueError('Old feature evidence changed')
    old=json.loads(old_path.read_text());extra=json.loads(extra_path.read_text())
    if extra['status']!='features_complete' or extra['contract_sha256']!=sha(ROOT/'CONTRACT.json'):raise ValueError('Wrong new feature contract')
    args.output.mkdir(exist_ok=False);members=[]
    for original in old['members']:
        added=next(m for m in extra['members'] if m['source']==original['source'])
        if added['encoders']!=original['encoders']:raise ValueError('Encoder change confounds data test')
        op=args.old/original['path'];ap=args.extra/added['path']
        if sha(op)!=original['sha256'] or sha(ap)!=added['sha256']:raise ValueError('Feature arrays changed')
        with np.load(op,allow_pickle=False) as old_packet,np.load(ap,allow_pickle=False) as extra_packet:
            if set(old_packet.files)!=set(extra_packet.files):raise ValueError('Feature schema changed')
            if not np.all(extra_packet['role']=='optimization') or not np.all(extra_packet['labels']==1) or len(extra_packet['labels'])!=21:
                raise ValueError('Only 21 new optimization positives allowed')
            merged={key:np.concatenate((old_packet[key],extra_packet[key])) for key in old_packet.files}
            count=len(old_packet['labels']);selection=merged['role']=='selection'
            for key in old_packet.files:
                np.testing.assert_array_equal(merged[key][:count],old_packet[key])
                np.testing.assert_array_equal(merged[key][selection],old_packet[key][old_packet['role']=='selection'])
            path=args.output/original['path'];np.savez_compressed(path,**merged)
            members.append(dict(source=original['source'],path=path.name,sha256=sha(path),bytes=path.stat().st_size,
                                rows=len(merged['labels']),columns=merged['features'].shape[1],encoders=original['encoders']))
    report=dict(status='features_complete',members=members,contract_sha256=sha(ROOT/'CONTRACT.json'),
                old_manifest_sha256=sha(old_path),extra_manifest_sha256=sha(extra_path),all_old_rows_unchanged=True,
                selection_exact=True,added_positive_optimization_rows=21,target_pilot_labels_used=False,authorized_for_submission=False)
    (args.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
