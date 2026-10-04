"""Verify complete additional-fitting raw artifacts before accessing labels."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_extra_fit_scope import scope
BASE=runpy.run_path(str(ROOT/'scripts/verify-focus-source-cache.py'))
RUN='focus-extra-fit-cache-v1'
NOTEBOOK_SHA='bed7e37bd8ab5537eb2b58ad87974952897d89d43c54fe9ee4b1a1d88584af96'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    if sha(notebook)!=NOTEBOOK_SHA:raise ValueError('Exact frozen additional-fitting notebook required')
    policy=scope((ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    if json.loads(notebook.read_text())['metadata']['codex']['contract']!=policy:raise ValueError('Fixed original-fitting-only scope changed')
    probe_path=ROOT/'reports/experiments/focus-source-probe-v1-result.json'
    if sha(probe_path)!=BASE['PROBE_SHA']:raise ValueError('Exact successful detector smoke required')
    probe=json.loads(probe_path.read_text());smoke={r['stem']:r for r in probe['records']}
    terminal_path=folder/'focus_extra_fit_cache_terminal.json';terminal=json.loads(terminal_path.read_text())
    expected=policy['replay_stems']+policy['training_stems']
    if (terminal['status']!='completed' or terminal['run_id']!=RUN or terminal['frozen_stems']!=expected
        or terminal['training_stems']!=policy['training_stems'] or terminal['split_sha256']!=policy['split_sha256']
        or terminal['model_sha256']!=BASE['MODEL_SHA'] or terminal['declared_budget_seconds']!=3600
        or not 0<terminal['elapsed_seconds']<=3600 or terminal['exact_smoke_replay'] is not True
        or terminal['ground_truth_opened'] is not False or terminal['new_target_movies_opened']!=0
        or terminal['submission_created'] is not False or [r['stem'] for r in terminal['records']]!=expected):
        raise ValueError('Complete bounded additional fitting cache required')
    if sha(folder/'verified_runtime_identity.json')!=probe['runtime_identity_sha256']:raise ValueError('Audited detector runtime changed')
    records=[]
    for record in terminal['records']:
        stem=record['stem'];frames=3 if stem in smoke else 100;path=folder/'raw_detections'/(stem+'.npz')
        if sha(path)!=record['sha256']:raise ValueError('Raw artifact checksum mismatch')
        if {k:v for k,v in record.items() if k!='scope'}!=json.loads(path.with_suffix('.json').read_text()):raise ValueError('Raw metadata differs from terminal')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'coords','movie_shape','scale_um'}:raise ValueError('Raw centroid schema changed')
            coords=data['coords'].copy();counts=BASE['validate_arrays'](coords,data['movie_shape'],data['scale_um'],frames)
        if (record['frame_counts']!=counts or record['node_count']!=len(coords) or record['failed_frames']!=0
            or record['postprocessing_applied'] is not False or record['ground_truth_opened'] is not False
            or record['scope']!=('training_smoke_replay' if stem in smoke else 'training_cache')
            or np.any(np.diff(coords[:,0])<0)):
            raise ValueError('All raw frames/nodes must be preserved in order')
        if stem in smoke:
            reference=ROOT/'.biohub/cache/kernel-outputs/focus-source-probe-v1/raw_detections'/(stem+'.npz')
            if sha(reference)!=smoke[stem]['sha256']:raise ValueError('Earlier six-frame reference changed')
            with np.load(reference,allow_pickle=False) as data:
                if not np.array_equal(coords,data['coords']):raise ValueError('New run failed exact prior detector replay')
        records.append(dict(stem=stem,role='replay' if stem in smoke else 'fitting',frames=frames,nodes=len(coords),max_frame_nodes=max(counts),sha256=record['sha256']))
    return dict(status='verified_raw_focus_extra_fitting_cache',run_id=RUN,contract=policy,records=records,
        terminal=terminal,terminal_sha256=sha(terminal_path),notebook_sha256=NOTEBOOK_SHA,
        complete_training_frames=800,smoke_frames_replayed=6,source_selection_opened=False,new_target_movies_opened=0,
        ground_truth_opened=False,authorized_for_submission=False,pretraining_overlap_verified=False)


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite verified experiment')
    result=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}')
    target.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='terminal'},indent=2))
