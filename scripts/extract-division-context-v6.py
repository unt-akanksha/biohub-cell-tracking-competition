"""New neighboring image observations; labels and original patches stay exact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import threading
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.native_correspondence_data_v2 import normalize_gpu,patches_gpu
from research.native_division_context_data_v6 import context_layout


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-proof',type=Path);args=parser.parse_args()
    helper=runpy.run_path(str(ROOT/'scripts/build-native-correspondence-v2.py'));helper['preflight']()
    cp=ROOT/'CONTRACT.json';contract=json.loads(cp.read_text())
    for r in contract['files']:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Context bundle changed')
    pp=ROOT/'plan/MOVIES.json';plan=json.loads(pp.read_text());private=json.loads((ROOT/'plan/PRIVATE_ARCHIVE_PLAN.json').read_text())
    if private['movie_plan_sha256']!=sha(pp):raise ValueError('Wrong temporal frame scope')
    sources={s['alias']:Path(s['root']) for s in plan['sources']}
    for s in plan['sources']:
        if sha(sources[s['alias']]/'DATA.json')!=s['data_sha256']:raise ValueError('Source labels/patches changed')
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/dev/shm')):raise ValueError('New owned RAM output required')
    if shutil.disk_usage('/dev/shm').free<3*1024**3:raise RuntimeError('Insufficient RAM storage reserve')
    if not args.smoke:
        proof=json.loads(args.smoke_proof.read_text())
        if proof['status']!='context_smoke_passed' or proof['contract_sha256']!=sha(cp):raise ValueError('Matching context smoke required')
        if proof['estimated_full_seconds']>1000:raise RuntimeError('Smoke projects insufficient watchdog headroom')
    chosen=plan['records']
    if args.smoke:
        chosen=list({r['id']:r for embryo in ('44b6','6bba') for label in ('positive','negative')
                     for r in [next(r for r in chosen if r['embryo']==embryo and r['role']=='optimization' and r[label]>0)]}.values())
    ids={r['id'] for r in chosen};by_id={r['id']:r for r in chosen}
    args.output.mkdir();(args.output/'data').mkdir();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
    started=time.monotonic();timer=threading.Timer(1200,lambda:os._exit(124));timer.daemon=True;timer.start()
    records=[];frames=[];totals={};bytes_written=0
    result=dict(status='running',contract_sha256=sha(cp),plan_sha256=sha(pp),records=records,frames=frames,totals=totals,
                old_data_unchanged=True,target_pilot_labels_used=False,authorized_for_submission=False)
    archive=helper['Archive'](private)
    try:
        for movie in plan['movies']:
            selected=[by_id[i] for i in movie['record_ids'] if i in ids]
            if not selected:continue
            helper['preflight']();packets={};layouts={};requests={}
            for r in selected:
                path=sources[r['source_alias']]/r['path']
                if sha(path)!=r['sha256'] or path.stat().st_size!=r['bytes']:raise ValueError('Source packet changed')
                with np.load(path,allow_pickle=False) as p:packets[r['id']]={k:p[k] for k in p.files}
                layout=context_layout(packets[r['id']],r['transition']);layouts[r['id']]=layout
                packets[r['id']]['context_patches']=np.zeros_like(packets[r['id']]['patches'])
                packets[r['id']]['context_valid']=layout['context_valid']
                packets[r['id']]['context_is_parent']=layout['is_parent']
                for frame_time in sorted(set(map(int,layout['context_times'][layout['context_valid']]))):
                    index=np.flatnonzero(layout['context_times']==frame_time)
                    requests.setdefault(frame_time,[]).append((r['id'],index,layout['positions'][index]))
            images,metadata=archive.movie(movie,sorted(requests))
            for frame_time,items in sorted(requests.items()):
                helper['preflight']();normalized,_=normalize_gpu(images[frame_time],torch)
                if 'geometry_max_error' not in result:result['geometry_max_error']=helper['geometry_smoke'](normalized)
                positions=np.concatenate([item[2] for item in items]);sampled=patches_gpu(normalized,positions,torch)
                first=0
                for rid,index,points in items:
                    packets[rid]['context_patches'][index]=sampled[first:first+len(index)];first+=len(index)
                frames.append(dict(stem=movie['stem'],time=frame_time,raw_sha256=hashlib.sha256(images[frame_time].tobytes()).hexdigest(),patches=len(positions)))
            for r in selected:
                packet=packets[r['id']];path=args.output/'data'/f'{r["id"]:04d}.npz'
                if not np.isfinite(packet['context_patches']).all():raise ValueError('Nonfinite temporal pixels')
                np.savez_compressed(path,**packet)
                with np.load(path,allow_pickle=False) as restored,np.load(sources[r['source_alias']]/r['path'],allow_pickle=False) as old:
                    for key in old.files:np.testing.assert_array_equal(restored[key],old[key])
                size=path.stat().st_size;bytes_written+=size
                if bytes_written>512*1024**2:raise RuntimeError('Temporal patch storage cap')
                records.append({**r,'path':path.relative_to(args.output).as_posix(),'sha256':sha(path),'bytes':size,
                                'source_packet_sha256':r['sha256'],'original_fields_exact':True,
                                'invalid_context_patches':int((~packet['context_valid']).sum())})
                total=totals.setdefault(r['embryo']+'-'+r['role'],dict(positive=0,negative=0,triples=0))
                for key in total:total[key]+=r[key]
            del images,packets,normalized
            helper['save'](args.output/'PROGRESS.json',result)
            print(json.dumps(dict(stem=movie['stem'],records=len(records),frames=len(frames),bytes=bytes_written,seconds=time.monotonic()-started)),flush=True)
        if not args.smoke and (len(records)!=len(plan['records']) or totals!=plan['totals']):raise ValueError('Incomplete unchanged example inventory')
        result['status']='context_smoke_passed' if args.smoke else 'context_data_complete'
    except Exception as error:result.update(status='failed',error_type=type(error).__name__)
    finally:
        timer.cancel();elapsed=time.monotonic()-started
        result.update(seconds=elapsed,bytes=bytes_written,estimated_full_seconds=elapsed/max(len(frames),1)*sum(len(m['image_frames']) for m in plan['movies']))
        helper['save'](args.output/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','frames')}),flush=True)
    return 0 if result['status'] in ('context_smoke_passed','context_data_complete') else 1


if __name__=='__main__':raise SystemExit(main())
