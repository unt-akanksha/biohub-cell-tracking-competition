"""Independently replay every saved smoke centroid before permitting expansion."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/build-focus-source-probe.py'))
NOTEBOOK_SHA = '1ad40fc75748219cc9f6b6b5a1deb8f14cfd67dfd8242ad18dcc0135cc85a6e7'


def verify(folder):
    notebook = ROOT / 'kaggle/biohub-focus-source-probe-v1/biohub-focus-source-probe-v1.ipynb'
    if hashlib.sha256(notebook.read_bytes()).hexdigest() != NOTEBOOK_SHA:
        raise ValueError('Frozen smoke notebook changed')
    terminal = json.loads((folder / 'focus_source_probe_terminal.json').read_text())
    if (terminal['status'] != 'completed' or terminal['run_id'] != M['RUN']
            or terminal['frozen_stems'] != M['STEMS'] or terminal['declared_budget_seconds'] != 3600
            or not 0 < terminal['elapsed_seconds'] <= 3600 or terminal['exact_raw_replay'] is not True
            or terminal['ground_truth_opened'] is not False or terminal['submission_created'] is not False
            or terminal['model_sha256'] != M['AUDIT']['EXPECTED_SHA']):
        raise ValueError('Complete exact six-frame replay required')
    identity = json.loads((folder / 'verified_runtime_identity.json').read_text())
    cpu = json.loads((ROOT / '.biohub/cache/kernel-outputs/focus-runtime-identity-v1/focus_runtime_identity.json').read_text())
    if identity['status'] != 'matched' or identity['files'] != cpu['files'] or identity['model_sha256'] != cpu['model_sha256']:
        raise ValueError('Exact CPU-audited runtime required')
    if [r['stem'] for r in terminal['records']] != M['STEMS']:
        raise ValueError('Smoke record scope changed')
    records = []
    for record in terminal['records']:
        stem = record['stem']
        path = folder / 'raw_detections' / f'{stem}.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256'] or record['frames'] != 3 or record['exact_replay'] is not True:
            raise ValueError('Actual smoke file checksum mismatch')
        old_path = ROOT / '.biohub/cache/kernel-outputs/focus3d-raw-detections-v1/raw_detections' / f'{stem}.npz'
        expected_old = {'6bba_f1fde7e0': 'fcf6eb4a0586b1c77e5eaa9221543e1ec196ffcee091932a81dc1974fc40bc8d',
                        '6bba_23af9eeb': '7d62559975797b50f9390ba68d2c76be358cd9c303b74459d1d871570111c7b2'}
        if hashlib.sha256(old_path.read_bytes()).hexdigest() != expected_old[stem]:
            raise ValueError('Original raw reference changed on host')
        with np.load(old_path, allow_pickle=False) as old:
            expected = old['coords'][old['coords'][:, 0] < 3]
        with np.load(path, allow_pickle=False) as current:
            if (not np.array_equal(current['coords'], expected)
                    or not np.array_equal(current['movie_shape'], [3, 64, 256, 256])
                    or not np.array_equal(current['scale_um'], [1.625, .40625, .40625])
                    or len(expected) != record['nodes']):
                raise ValueError('Host replay differs from original raw detections')
        records.append(record)
    return dict(status='verified_six_frame_focus_source_probe', terminal=terminal,
                records=records, notebook_sha256=NOTEBOOK_SHA,
                runtime_identity_sha256=hashlib.sha256((folder / 'verified_runtime_identity.json').read_bytes()).hexdigest(),
                cpu_exact_replay=True, independent_accuracy_established=False,
                authorized_for_submission=False)


if __name__ == '__main__':
    target = ROOT / 'reports/experiments/focus-source-probe-v1-result.json'
    if target.exists():
        raise ValueError('Refuse to overwrite verified source smoke')
    report = verify(ROOT / '.biohub/cache/kernel-outputs/focus-source-probe-v1')
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
