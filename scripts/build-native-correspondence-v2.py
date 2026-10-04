"""Bounded selected-frame extraction; explicit smoke proof before a full build."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
import numcodecs
import requests
import torch
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.kaggle_archive_ranges import extract_member
from research.native_correspondence_data_v2 import VOXEL, SCALES, groups, pack_groups, normalize_gpu, patches_gpu, proposals
from research.visual_correspondence_data_v1 import validate_roles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def preflight():
    response = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True, timeout=10)
    if any(int(line.strip()) != os.getpid() for line in response.stdout.splitlines() if line.strip()):
        raise RuntimeError('Foreign GPU process: yield without interrupting it')
    memory = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if memory['MemAvailable'] < 3*1024**3:
        raise RuntimeError('System RAM reserve reached')


class Archive:
    def __init__(self, plan):
        self.plan = plan; self.local = threading.local()
        self.records = {r['path']: r for r in plan['records']}

    def member(self, name):
        record = self.records[name]
        first = record['header_offset']
        last = min(self.plan['archive_bytes']-1, first+record['compressed_bytes']+131102-1)
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        for attempt in range(3):
            response = self.local.session.get(self.plan['archive_url'], headers={
                'Range': f'bytes={first}-{last}', 'If-Match': self.plan['archive_etag'],
                'Accept-Encoding': 'identity'}, timeout=(10, 40))
            try:
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{self.plan["archive_bytes"]}':
                    raise ValueError('Pinned range not honored')
                data = response.content
                if len(data) != last-first+1:
                    raise ValueError('Incomplete range')
                def fetch(a, b):
                    if not first <= a <= b <= last:
                        raise ValueError('ZIP header exceeds bounded prefetched range')
                    return data[a-first:b-first+1]
                return extract_member(record, fetch)
            except (ValueError, requests.RequestException):
                if attempt == 2:
                    raise
            finally:
                response.close()

    def movie(self, movie, frames):
        prefix = 'train/'+movie['stem']+'.zarr/'
        group_bytes = self.member(prefix+'zarr.json')
        array_bytes = self.member(prefix+'0/zarr.json')
        array = json.loads(array_bytes)
        if array['shape'] != [100,64,256,256] or array['data_type'] != 'uint16' or array['chunk_grid']['configuration']['chunk_shape'] != [1,64,256,256]:
            raise ValueError('Native array contract changed')
        codecs = array['codecs']
        if len(codecs) != 2 or codecs[0] != {'name':'bytes','configuration':{'endian':'little'}} or codecs[1]['name'] != 'blosc':
            raise ValueError('Unsupported native codec')
        group = json.loads(group_bytes)
        transforms = group['attributes']['multiscales'][0]['datasets'][0]['coordinateTransformations']
        scale = next(t['scale'] for t in transforms if t['type']=='scale')
        if not np.allclose(scale[-3:], VOXEL, rtol=0, atol=1e-8):
            raise ValueError('Native physical units changed')
        def frame(t):
            encoded = self.member(prefix+f'0/c/{t}/0/0/0')
            decoded = numcodecs.Blosc().decode(encoded)
            if len(decoded) != 64*256*256*2:
                raise ValueError('Native decoded length mismatch')
            return t, np.frombuffer(decoded, '<u2').reshape(64,256,256)
        with ThreadPoolExecutor(max_workers=4) as pool:
            images = dict(pool.map(frame, frames))
        return images, dict(group_sha256=hashlib.sha256(group_bytes).hexdigest(), array_sha256=hashlib.sha256(array_bytes).hexdigest())


def geometry_smoke(image):
    points = np.array([[0,0,0],[31.25,46.125,62.875],[102.375,103.59375,103.59375]], np.float32)
    actual = patches_gpu(image, points, torch).astype(np.float32)
    axis = np.arange(-7,8, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'))
    reference = np.empty_like(actual)
    cpu = image.cpu().numpy()
    for row, point in enumerate(points):
        for channel, scale in enumerate(SCALES):
            reference[row,channel] = map_coordinates(cpu, (point[:,None,None,None]+grid*scale)/VOXEL[:,None,None,None],
                                                        order=1, mode='nearest', prefilter=False)
    error = float(np.max(np.abs(actual-reference)))
    if error > .002:
        raise ValueError('GPU/CPU physical crop mismatch')
    return error


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-proof', type=Path); args=parser.parse_args()
    begin=time.monotonic(); preflight()
    if not args.output.resolve().is_relative_to(Path('/dev/shm')) or args.output.exists():
        raise ValueError('Require a new owned output under /dev/shm')
    if shutil.disk_usage('/dev/shm').free < 4*1024**3:
        raise ValueError('Insufficient temporary RAM storage')
    movie_path=args.plan/'MOVIES.json'; archive_path=args.plan/'PRIVATE_ARCHIVE_PLAN.json'
    manifest_path=args.plan/'CODE.json'; manifest=json.loads(manifest_path.read_text())
    for record in manifest['files']:
        if sha(ROOT/record['path']) != record['sha256']:
            raise ValueError('Executable code changed')
    movies=json.loads(movie_path.read_text()); archive_plan=json.loads(archive_path.read_text())
    if sha(movie_path) != '60300fbc7251f842e80b3ae1687e78d097b7dcda876908f3832b34174c0a0ef6' or archive_plan['movie_plan_sha256'] != sha(movie_path):
        raise ValueError('Movie plan changed')
    validate_roles(movies['movies'])
    if not args.smoke:
        proof=json.loads(args.smoke_proof.read_text())
        if proof['status'] != 'smoke_passed' or proof['code_sha256'] != sha(manifest_path) or proof['movie_plan_sha256'] != sha(movie_path):
            raise ValueError('Full build requires this exact passed smoke')
    selected=movies['movies']
    if args.smoke:
        selected=[next(m for m in selected if m['embryo']==e and m['role']=='optimization') for e in ('44b6','6bba')]
    args.output.mkdir(); (args.output/'data').mkdir()
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.85)
    timer=threading.Timer(600 if args.smoke else 2700, lambda: os._exit(124)); timer.daemon=True; timer.start()
    records=[]; completed=[]; counters={}; bytes_written=0; archive=Archive(archive_plan)
    state=dict(run_id='native-correspondence-v2', status='running', smoke=args.smoke,
               code_sha256=sha(manifest_path), movie_plan_sha256=sha(movie_path), archive_plan_sha256=sha(archive_path),
               patch_scales_um=list(SCALES), competition_test_data_read=False, sealed_audit_opened=False,
               authorized_for_submission=False, records=records, completed_movies=completed, totals=counters)
    try:
        for movie in selected:
            preflight(); stem=movie['stem']; movie_start=time.monotonic()
            times=movie['selected_transitions'][:4] if args.smoke else movie['selected_transitions']
            frames=sorted({f for t in times for f in (t,t+1)})
            images, metadata=archive.movie(movie, frames)
            nodes={int(n[0]):n for n in movie['nodes']}; cached=OrderedDict()
            key=movie['embryo']+'-'+movie['role']
            total=counters.setdefault(key, dict(annotated_edges=0,matched_queries=0,positive=0,null=0,ambiguous_omitted=0,eligible=0))
            for t in times:
                preflight()
                for f in (t,t+1):
                    if f not in cached:
                        image, pooled=normalize_gpu(images[f], torch)
                        if args.smoke and not state.get('geometry_max_error'):
                            state['geometry_max_error']=geometry_smoke(image)
                        cached[f]=(image, proposals(pooled))
                        while len(cached)>2:
                            cached.popitem(last=False)
                edges=[e for e in movie['edges'] if int(nodes[e[0]][1])==t]
                parents=np.array([nodes[e[0]][2:] for e in edges], np.float32)*VOXEL
                children=np.array([nodes[e[1]][2:] for e in edges], np.float32)*VOXEL
                pimage, pcoords=cached[t]; cimage, ccoords=cached[t+1]
                labels, counts=groups(pcoords, ccoords, parents, children)
                for name,value in counts.items(): total[name]+=value
                if not labels:
                    continue
                packet, ppoints, cpoints=pack_groups(labels,pcoords,ccoords)
                packet['patches']=np.concatenate((patches_gpu(pimage,ppoints,torch), patches_gpu(cimage,cpoints,torch)))
                if not np.isfinite(packet['patches']).all():
                    raise ValueError('Nonfinite patches')
                relative=f'data/{stem}-{t:03d}.npz'; path=args.output/relative
                if bytes_written+packet['patches'].nbytes > 3*1024**3:
                    raise RuntimeError('Temporary storage cap reached: harvest before continuing')
                np.savez_compressed(path, **packet)
                size=path.stat().st_size; bytes_written+=size
                records.append(dict(path=relative, sha256=sha(path), bytes=size, stem=stem, embryo=movie['embryo'],
                                    role=movie['role'], transition=t, groups=len(labels), patches=len(packet['patches']), counts=counts))
            del cached, images
            completed.append(dict(stem=stem, transitions=len(times), seconds=time.monotonic()-movie_start, **metadata))
            state.update(elapsed_seconds=time.monotonic()-begin, bytes=bytes_written)
            save(args.output/'PROGRESS.json',state)
            print(json.dumps(dict(status='movie_complete', stem=stem, movies=len(completed), packets=len(records), bytes=bytes_written, seconds=state['elapsed_seconds'])), flush=True)
        if args.smoke and (len(completed)!=2 or any(v['positive']==0 for v in counters.values())):
            raise ValueError('Real-source smoke lacks positive correspondences')
        state['status']='smoke_passed' if args.smoke else 'data_complete'
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__)
    finally:
        state.update(elapsed_seconds=time.monotonic()-begin, bytes=bytes_written)
        save(args.output/'RESULT.json',state); timer.cancel()
        print(json.dumps({k:v for k,v in state.items() if k not in ('records','completed_movies')}), flush=True)
    return 0 if state['status'] in ('smoke_passed','data_complete') else 1


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(status='preflight_failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1)
