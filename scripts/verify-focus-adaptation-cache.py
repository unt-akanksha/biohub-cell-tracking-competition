"""Verify the complete new training-only detector cache before label access."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_adaptation_cache_contract import scope,SPLIT_SHA
BASE=runpy.run_path(str(ROOT/'scripts/verify-focus-source-cache.py'))
RUN='focus-adaptation-cache-v1'
NOTEBOOK_SHA='efd1cdef0e762c8ed7bb3ce045365dfbfd854c04a3dbe79e0ef99d969a20a8bd'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    if sha(notebook)!=NOTEBOOK_SHA:raise ValueError('Exact frozen adaptation cache notebook required')
    policy=scope((ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    nb=json.loads(notebook.read_text())
    if nb['metadata']['codex']['contract']!=policy:raise ValueError('Training-only movie contract changed')
    probe_path=ROOT/'reports/experiments/focus-source-probe-v1-result.json'
    if sha(probe_path)!=BASE['PROBE_SHA']:raise ValueError('Frozen successful detector probe required')
    probe=json.loads(probe_path.read_text());smoke={r['stem']:r for r in probe['records']}
    expected=policy['replay_stems']+policy['training_stems']
    terminal_path=folder/'focus_adaptation_cache_terminal.json';terminal=json.loads(terminal_path.read_text())
    if (terminal['status']!='completed' or terminal['run_id']!=RUN or terminal['frozen_stems']!=expected
        or terminal['training_stems']!=policy['training_stems'] or terminal['split_sha256']!=SPLIT_SHA
        or terminal['model_sha256']!=BASE['MODEL_SHA'] or terminal['declared_budget_seconds']!=3600
        or not 0<terminal['elapsed_seconds']<=3600 or terminal['exact_smoke_replay'] is not True
        or terminal['ground_truth_opened'] is not False or terminal['new_target_movies_opened']!=0
        or terminal['submission_created'] is not False or [r['stem'] for r in terminal['records']]!=expected):
        raise ValueError('Complete bounded training cache and exact replay required')
    if sha(folder/'verified_runtime_identity.json')!=probe['runtime_identity_sha256']:raise ValueError('Audited detector runtime identity changed')
    records=[]
    for record in terminal['records']:
        stem=record['stem'];frames=3 if stem in smoke else 100;path=folder/'raw_detections'/(stem+'.npz')
        if sha(path)!=record['sha256']:raise ValueError('Raw cache artifact checksum mismatch')
        metadata=json.loads(path.with_suffix('.json').read_text())
        if {k:v for k,v in record.items() if k!='scope'}!=metadata:raise ValueError('Raw metadata differs from terminal')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'coords','movie_shape','scale_um'}:raise ValueError('Raw centroid schema changed')
            coords=data['coords'].copy();counts=BASE['validate_arrays'](coords,data['movie_shape'],data['scale_um'],frames)
        if (record['frame_counts']!=counts or record['node_count']!=len(coords) or record['failed_frames']!=0
            or record['postprocessing_applied'] is not False or record['ground_truth_opened'] is not False
            or record['scope']!=('training_smoke_replay' if stem in smoke else 'training_cache')
            or np.any(np.diff(coords[:,0])<0)):
            raise ValueError('Complete ordered unprocessed frame coverage required')
        if stem in smoke:
            path=ROOT/'.biohub/cache/kernel-outputs/focus-source-probe-v1/raw_detections'/(stem+'.npz')
            if sha(path)!=smoke[stem]['sha256']:raise ValueError('Original small probe changed')
            with np.load(path,allow_pickle=False) as data:
                if not np.array_equal(coords,data['coords']):raise ValueError('Actual detector smoke replay differs')
        role='replay' if stem in smoke else 'fitting' if stem in policy['fitting_stems'] else 'diagnostic'
        records.append(dict(stem=stem,role=role,frames=frames,nodes=len(coords),max_frame_nodes=max(counts),sha256=record['sha256']))
    return dict(status='verified_raw_focus_adaptation_cache',run_id=RUN,contract=policy,records=records,
        terminal=terminal,terminal_sha256=sha(terminal_path),notebook_sha256=NOTEBOOK_SHA,
        complete_training_frames=800,smoke_frames_replayed=6,source_selection_opened=False,new_target_movies_opened=0,
        ground_truth_opened=False,authorized_for_submission=False,pretraining_overlap_verified=False)


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite verified result')
    report=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}')
    target.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='terminal'},indent=2))
