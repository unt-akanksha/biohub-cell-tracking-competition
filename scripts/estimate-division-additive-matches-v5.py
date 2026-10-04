"""Dry-run additive supervision over existing source-only cached proposals."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import VOXEL,match_queries
from research.native_division_matches_v5 import isolated_additive_matches


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'.biohub/cache/native-division-proposal-audit-v4-full-output'
    result_path=folder/'RESULT.json'
    if sha(result_path)!='0f12f8a011c0575f63f9160e97db2dcc2dc8ccab6cb8c898e3963cc18e6a2d4e':raise ValueError('Audit changed')
    result=json.loads(result_path.read_text())
    plan_path=ROOT/'.biohub/cache/native-division-proposal-audit-v4-plan/MOVIES.json'
    if sha(plan_path)!=result['movie_plan_sha256']:raise ValueError('Source scope changed')
    movies={m['stem']:m for m in json.loads(plan_path.read_text())['movies']}
    totals={};events=[]
    for artifact in result['point_artifacts']:
        path=folder/artifact['path']
        if sha(path)!=artifact['sha256']:raise ValueError('Point cache changed')
        movie=movies[path.stem];count=totals.setdefault(movie['embryo'],Counter())
        if movie['role']!='optimization':raise ValueError('Source only')
        mappings={};points={};nodes={int(n[0]):n for n in movie['nodes']}
        with np.load(path,allow_pickle=False) as packet:
            for t in movie['image_frames']:
                points[t]=packet[f'baseline_{t:03d}']
                ns=[n for n in movie['nodes'] if int(n[1])==t]
                coords=np.array([n[2:] for n in ns],np.float32).reshape(-1,3)*VOXEL
                mappings[t]={name:{int(ns[i][0]):j for i,j in matcher(coords,points[t]).items()}
                             for name,matcher in [('strict',match_queries),('additive',isolated_additive_matches)]}
            outgoing={}
            for p,c in movie['edges']:outgoing.setdefault(p,[]).append(c)
            for p,children in outgoing.items():
                if len(children)!=2:continue
                t=int(nodes[p][1]);out={}
                for name in ('strict','additive'):
                    pm=mappings[t][name];cm=mappings[t+1][name]
                    out[name]=p in pm and all(c in cm for c in children)
                    if out[name]:out[name]=all(np.linalg.norm(points[t][pm[p]]-points[t+1][cm[c]])<=20 for c in children)
                    out[name]=bool(out[name])
                if out['strict'] and not out['additive']:raise ValueError('Lost prior eligible event')
                count.update(events=1,strict=int(out['strict']),additive=int(out['additive']),newly_available=int(out['additive'] and not out['strict']))
                events.append(dict(stem=movie['stem'],parent=p,children=children,transition=t,**out))
    report=dict(status='source_additive_match_dry_run',totals={k:dict(v) for k,v in totals.items()},events=events,
                matcher_sha256=sha(ROOT/'research/native_division_matches_v5.py'),source_result_sha256=sha(result_path),
                canonical_training_labels_changed=False,selection_opened=False,target_pilot_labels_used=False,
                authorized_for_submission=False)
    (ROOT/'reports/experiments/native-division-additive-matches-v5-dry-run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='events'}))


if __name__=='__main__':main()
