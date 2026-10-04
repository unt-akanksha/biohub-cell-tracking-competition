"""Hash-bound, unchanged labeled examples with extra image-frame requests only."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_division_context_data_v6 import context_layout


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources=[('legacy',ROOT/'.biohub/cache/native-division-v3-data-bundle','/dev/shm/biohub-native-division-v3-data-bundle',
              '7e0e7b1ca3c11b27319824fad3746b49f8a1cf519a73d3dad38e12d2ef8a0e06'),
             ('additive',ROOT/'.biohub/cache/native-division-additive-v5-full-output','/dev/shm/biohub-native-division-additive-v5-full',
              'e358a984eade85b96fbf2d77937ba0d57307f4813f44fb6c339346024d17df87')]
    prior=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    if sha(prior)!='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6':raise ValueError('Source/selection roles changed')
    roles={m['stem']:(m['embryo'],m['role']) for m in json.loads(prior.read_text())['movies']}
    records=[];movies={};totals={};provenance=[];patches=0;invalid=0
    for alias,local,remote,expected in sources:
        path=local/'DATA.json'
        if sha(path)!=expected:raise ValueError('Source dataset changed')
        data=json.loads(path.read_text());provenance.append(dict(alias=alias,root=remote,data_sha256=expected))
        for r in data['records']:
            if roles.get(r['stem'])!=(r['embryo'],r['role']):raise ValueError('Record outside fixed split')
            packet_path=local/r['path']
            if sha(packet_path)!=r['sha256']:raise ValueError('Old packet changed')
            with np.load(packet_path,allow_pickle=False) as p:
                layout=context_layout(p,r['transition']);patches+=len(p['patches']);invalid+=int((~layout['context_valid']).sum())
                frame_ids=sorted(set(map(int,layout['context_times'][layout['context_valid']])))
                if len(p['labels'])!=r['triples'] or int(p['labels'].sum())!=r['positive']:raise ValueError('Labels changed')
            record={**r,'source_alias':alias,'id':len(records),'context_frames':frame_ids};records.append(record)
            movie=movies.setdefault(r['stem'],dict(stem=r['stem'],embryo=r['embryo'],role=r['role'],image_frames=set(),record_ids=[]))
            movie['image_frames'].update(frame_ids);movie['record_ids'].append(record['id'])
            count=totals.setdefault(r['embryo']+'-'+r['role'],dict(positive=0,negative=0,triples=0))
            for key in count:count[key]+=r[key]
    for movie in movies.values():movie['image_frames']=sorted(movie['image_frames'])
    if sum(v['triples'] for v in totals.values())!=3108:raise ValueError('Whole v5 dataset required')
    output=ROOT/'.biohub/cache/native-division-context-v6-plan';output.mkdir(exist_ok=False)
    plan=dict(run_id='native-division-context-v6',movies=list(movies.values()),records=records,sources=provenance,totals=totals,
              patches=patches,invalid_boundary_patches=invalid,competition_test_data_read=False,sealed_audit_opened=False,
              authorized_for_submission=False,source_role_sha256=sha(prior))
    (output/'MOVIES.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(status='context_frames_frozen',movies=len(movies),records=len(records),patches=patches,
                         frames=sum(len(m['image_frames']) for m in movies.values()),invalid_boundary_patches=invalid,
                         totals=totals,movie_plan_sha256=sha(output/'MOVIES.json'))))


if __name__=='__main__':main()
