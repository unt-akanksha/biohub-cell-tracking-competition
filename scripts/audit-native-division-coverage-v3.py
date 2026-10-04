"""Inventory unused optimization-time division labels without opening audit movies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import zarr

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.monotonic();plan=ROOT/'.biohub/cache/native-correspondence-v2-plan/MOVIES.json'
    if sha(plan)!='60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6':raise ValueError('Native roles changed')
    movie_plan=json.loads(plan.read_text());root=ROOT/'.biohub/cache/competition-train-geffs-packed-v1'
    manifest=root/'train_geff_cache_manifest.json'
    if sha(manifest)!='744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9':raise ValueError('Official labels changed')
    records=[];totals={}
    for movie in movie_plan['movies']:
        # Existing source selection is unchanged; no new validation selection.
        stem=movie['stem'];folder=root/'train'/(stem+'.geff')
        for check in movie['geff_files']:
            p=root/'train'/check['relative_path']
            if p.stat().st_size!=check['bytes'] or sha(p)!=check['sha256']:raise ValueError('GEFF hash mismatch')
        g=zarr.open_group(str(folder),mode='r');ids=np.asarray(g['nodes/ids']).astype(np.int64)
        times=np.asarray(g['nodes/props/t/values']).astype(np.int64)
        edges=np.asarray(g['edges/ids']).astype(np.int64);time_by_id=dict(zip(map(int,ids),map(int,times)))
        degree=Counter(map(int,edges[:,0]));events=[]
        for parent,count in degree.items():
            if count!=2:continue
            t=time_by_id[parent];children=sorted(map(int,edges[edges[:,0]==parent,1]))
            if any(time_by_id[c]!=t+1 for c in children):raise ValueError('Nonadjacent annotated division')
            events.append(dict(parent=parent,children=children,time=t,in_native_v2=t in movie['selected_transitions']))
        record=dict(stem=stem,embryo=movie['embryo'],role=movie['role'],events=events,
                    total_events=len(events),v2_events=sum(e['in_native_v2'] for e in events),
                    missing_event_times=sorted({e['time'] for e in events if not e['in_native_v2']}))
        records.append(record);key=movie['embryo']+'-'+movie['role']
        total=totals.setdefault(key,dict(movies=0,movies_with_divisions=0,total_events=0,v2_events=0,additional_event_transitions=0))
        total['movies']+=1;total['movies_with_divisions']+=bool(events);total['total_events']+=len(events)
        total['v2_events']+=record['v2_events'];total['additional_event_transitions']+=len(record['missing_event_times'])
    result=dict(status='verified_division_coverage_inventory',totals=totals,movies=records,
        movie_plan_sha256=sha(plan),source_sha256=sha(Path(__file__)),optimization_and_existing_selection_only=True,
        sealed_audit_opened=False,additional_images_downloaded=False,training_launched=False,
        authorized_for_submission=False,elapsed_seconds=time.monotonic()-start)
    output=ROOT/'reports/experiments/native-division-coverage-v3.json'
    if output.exists():raise ValueError('Preserve existing inventory')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='movies'}))


if __name__=='__main__':main()
