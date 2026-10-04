"""Freeze only the 93 previously unsampled optimization division transitions."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import numpy as np
import zarr

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-optimization-divisions',action='store_true');args=parser.parse_args()
    start=time.monotonic();audit=ROOT/'reports/experiments/native-division-coverage-v3.json'
    if sha(audit)!='5e674aad7607930cbb2f0e96d204d8bd13931a6fd3f407ddea1bc8d0e49f8cbe':raise ValueError('Division inventory changed')
    old=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    if sha(old)!='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6':raise ValueError('Roles changed')
    previous={m['stem']:m for m in json.loads(old.read_text())['movies']}
    geffs=ROOT/'.biohub/cache/competition-train-geffs-packed-v1/train';movies=[]
    for item in json.loads(audit.read_text())['movies']:
        times=sorted({e['time'] for e in item['events']}) if args.all_optimization_divisions else item['missing_event_times']
        if item['role']!='optimization' or not times:continue
        stem=item['stem'];parent=previous[stem]
        for r in parent['geff_files']:
            path=geffs/r['relative_path']
            if sha(path)!=r['sha256'] or path.stat().st_size!=r['bytes']:raise ValueError('GT file changed')
        frames=sorted({f for t in times for f in (t,t+1)})
        g=zarr.open_group(str(geffs/(stem+'.geff')),mode='r');ids=np.asarray(g['nodes/ids']).astype(np.int64)
        nt=np.asarray(g['nodes/props/t/values']).astype(np.int64)
        coords=np.stack([np.asarray(g[f'nodes/props/{a}/values']) for a in ('z','y','x')],axis=1)
        keep=np.isin(nt,frames);nodes=[[int(i),int(t),*map(float,c)] for i,t,c in zip(ids[keep],nt[keep],coords[keep])]
        lookup=dict(zip(map(int,ids),map(int,nt)))
        edges=[[int(a),int(b)] for a,b in np.asarray(g['edges/ids']) if lookup[int(a)] in times]
        movies.append(dict(stem=stem,embryo=item['embryo'],role='optimization',selected_transitions=times,
                           image_frames=frames,nodes=nodes,edges=edges,geff_files=parent['geff_files']))
    expected=112 if args.all_optimization_divisions else 93
    name='native-division-proposal-audit-v4' if args.all_optimization_divisions else 'native-division-v3-extra'
    if sum(len(m['selected_transitions']) for m in movies)!=expected:raise ValueError('Exact frozen transition count required')
    output=ROOT/'.biohub/cache'/(name+'-plan');output.mkdir(exist_ok=False)
    result=dict(run_id=name,movies=movies,competition_test_data_read=False,sealed_audit_opened=False,
                authorized_for_submission=False,prior_movie_plan_sha256=sha(old),division_inventory_sha256=sha(audit),
                total_transitions=expected,total_frames=sum(len(m['image_frames']) for m in movies),elapsed_seconds=time.monotonic()-start)
    (output/'MOVIES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='extra_division_frames_frozen',movies=len(movies),frames=result['total_frames'],
                         movie_plan_sha256=sha(output/'MOVIES.json'),seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
