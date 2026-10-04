"""Recover only unambiguous GT identities in frozen native v2 image packets."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_division_data_v3 import legacy_bindings,triplets


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic();data=ROOT/'.biohub/cache/native-correspondence-v2-data'
    manifest=data/'RESULT.json';plan=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    if sha(manifest)!='f2861ce9f21307507ed521716ca3bf9a7adf3b76ce9f8216e4407233616c693a' or sha(plan)!='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6':raise ValueError('Frozen data/roles changed')
    payload=json.loads(manifest.read_text());movies={m['stem']:m for m in json.loads(plan.read_text())['movies']}
    output=ROOT/'.biohub/cache/native-division-v3-legacy';output.mkdir(exist_ok=False);(output/'data').mkdir()
    records=[];totals={};processed=[]
    for record in payload['records']:
        path=data/record['path']
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('Native packet changed')
        movie=movies[record['stem']]
        if movie['role']!=record['role']:raise ValueError('Role mismatch')
        with np.load(path,allow_pickle=False) as arrays:
            packet={k:arrays[k] for k in ('ids','coords','targets')}
            bindings,positions,counts=legacy_bindings(packet,movie,record['transition'])
            result,used,counts2=triplets(bindings,positions);counts.update(counts2)
            if used:result['patches']=arrays['patches'][used]
        key=record['embryo']+'-'+record['role'];total=totals.setdefault(key,{k:0 for k in counts})
        for k,v in counts.items():total[k]+=v
        processed.append(dict(stem=record['stem'],transition=record['transition'],counts=counts))
        if not used:continue
        name=Path(record['path']).name;destination=output/'data'/name
        np.savez_compressed(destination,**result)
        records.append(dict(path='data/'+name,sha256=sha(destination),bytes=destination.stat().st_size,
                            stem=record['stem'],embryo=record['embryo'],role=record['role'],transition=record['transition'],
                            source_packet_sha256=record['sha256'],patches=len(used),**counts2))
    result=dict(status='legacy_triplets_complete',run_id='native-division-v3',records=records,totals=totals,
        processed_packets=processed,native_data_sha256=sha(manifest),movie_plan_sha256=sha(plan),
        source_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'research/native_division_data_v3.py'),
        sealed_audit_opened=False,target_pilot_labels_used=False,authorized_for_submission=False,elapsed_seconds=time.monotonic()-start)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','totals','elapsed_seconds')}))


if __name__=='__main__':main()
