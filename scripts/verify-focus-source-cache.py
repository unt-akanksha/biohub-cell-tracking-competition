"""Validate complete source detections and exact smoke replay, without labels."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = 'focus-source-cache-v1'
NOTEBOOK_SHA = 'd65eac4ce6354829b13fd92176c3163b45a50efabe2c326f5b5322d7d73f322b'
SPLIT_SHA = '12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'
PROBE_SHA = '02a85eb8740981ca76635385ce237847aade1a8c960af7bbe78ee2e3366ef42b'
MODEL_SHA = 'b14a7bd272f824adb1a1073bc3f2af17a95919d5a0c3f1d9011a8d82378d8f3a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_arrays(coords, shape, scale, frames):
    if (coords.ndim != 2 or coords.shape[1] != 4 or coords.dtype != np.float32
            or not np.array_equal(shape, [frames, 64, 256, 256])
            or not np.array_equal(scale, [1.625, .40625, .40625])
            or not np.isfinite(coords).all() or np.any(coords < 0)
            or np.any(coords >= np.asarray([frames, 64, 256, 256]))
            or np.any(coords[:, 0] != np.floor(coords[:, 0]))):
        raise ValueError('Finite original in-bounds TZYX float32 centroids required')
    return [int(np.sum(coords[:, 0] == t)) for t in range(frames)]


def verify(folder):
    notebook = ROOT / f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    split = ROOT / 'research/independent_real_baseline_v1_split.json'
    probe_file = ROOT / 'reports/experiments/focus-source-probe-v1-result.json'
    if sha(notebook) != NOTEBOOK_SHA or sha(split) != SPLIT_SHA or sha(probe_file) != PROBE_SHA:
        raise ValueError('Exact frozen notebook, split and successful smoke required')
    fold = json.loads(split.read_text())['folds'][0]
    probe = json.loads(probe_file.read_text())
    smoke = {r['stem']: r for r in probe['records']}
    expected_stems = list(smoke) + fold['selection']
    terminal_file = folder / 'focus_source_cache_terminal.json'
    terminal = json.loads(terminal_file.read_text())
    if (terminal['status'] != 'completed' or terminal['run_id'] != RUN
            or terminal['frozen_stems'] != expected_stems or terminal['selection_stems'] != fold['selection']
            or terminal['split_sha256'] != SPLIT_SHA or terminal['model_sha256'] != MODEL_SHA
            or terminal['declared_budget_seconds'] != 3600 or not 0 < terminal['elapsed_seconds'] <= 3600
            or terminal['exact_smoke_replay'] is not True or terminal['ground_truth_opened'] is not False
            or terminal['new_target_movies_opened'] != 0 or terminal['submission_created'] is not False
            or [r['stem'] for r in terminal['records']] != expected_stems):
        raise ValueError('Complete bounded source cache and smoke replay required')
    identity_file = folder / 'verified_runtime_identity.json'
    if sha(identity_file) != probe['runtime_identity_sha256']:
        raise ValueError('Runtime identity changed from successful GPU smoke')
    records = []
    for record in terminal['records']:
        stem = record['stem']
        frames = 3 if stem in smoke else 100
        path = folder / 'raw_detections' / f'{stem}.npz'
        if sha(path) != record['sha256']:
            raise ValueError('Raw source artifact checksum mismatch')
        metadata = json.loads(path.with_suffix('.json').read_text())
        if {k: v for k, v in record.items() if k != 'scope'} != metadata:
            raise ValueError('Raw metadata differs from completed terminal')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'coords', 'movie_shape', 'scale_um'}:
                raise ValueError('Exact raw centroid schema required')
            coords = data['coords'].copy()
            counts = validate_arrays(coords, data['movie_shape'], data['scale_um'], frames)
        if (record['frame_counts'] != counts or record['node_count'] != len(coords)
                or record['failed_frames'] != 0 or record['postprocessing_applied'] is not False
                or record['ground_truth_opened'] is not False
                or record['scope'] != ('training_smoke_replay' if stem in smoke else 'source_selection')):
            raise ValueError('Exact complete-frame source coverage required')
        if stem in smoke:
            original = ROOT / '.biohub/cache/kernel-outputs/focus-source-probe-v1/raw_detections' / f'{stem}.npz'
            if sha(original) != smoke[stem]['sha256']:
                raise ValueError('Original successful smoke changed')
            with np.load(original, allow_pickle=False) as data:
                if not np.array_equal(coords, data['coords']):
                    raise ValueError('Host source/smoke replay differs')
        records.append(dict(stem=stem, frames=frames, nodes=len(coords), sha256=record['sha256'], scope=record['scope']))
    return dict(status='verified_raw_focus_source_cache', run_id=RUN, records=records,
                source_stems=fold['selection'], terminal=terminal, terminal_sha256=sha(terminal_file),
                notebook_sha256=NOTEBOOK_SHA, model_sha256=MODEL_SHA, split_sha256=SPLIT_SHA,
                complete_source_frames=800, smoke_frames_replayed=6, ground_truth_opened=False,
                authorized_for_submission=False, pretraining_overlap_verified=False)


if __name__ == '__main__':
    target = ROOT / f'reports/experiments/{RUN}-result.json'
    if target.exists():
        raise ValueError('Refuse to overwrite verified cache')
    report = verify(ROOT / f'.biohub/cache/kernel-outputs/{RUN}')
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'terminal'}, indent=2))
