"""Reuse previously prepared training-only arrays to measure resident throughput."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_pair_appearance_resident import Resident
from research.focus_pair_appearance_head import objective

PRIOR = runpy.run_path(str(ROOT / 'scripts/profile-focus-pair-appearance-prepared.py'))
sha = PRIOR['sha']


def main():
    started = time.monotonic()
    target = ROOT / 'reports/experiments/focus-pair-appearance-resident-v1-profile.json'
    if target.exists():
        raise ValueError('Never overwrite a completed resident profile')
    prior_path = ROOT / 'reports/experiments/focus-pair-appearance-prepared-v1-profile.json'
    prior = json.loads(prior_path.read_text())
    if prior['status'] != 'verified_prepared_cpu_equivalence_and_first_fold_profile':
        raise ValueError('Completed exact preparation smoke/profile required')
    if any(sha(ROOT / p) != v for p, v in prior['source_hashes'].items()):
        raise ValueError('Prepared method changed')
    evidence, samples, moments = PRIOR['load']()
    train = prior['training_stems']
    if train != [r['stem'] for r in evidence['records'][1:]]:
        raise ValueError('Exact first-fold fitting-only data required')
    training = [samples[s] for s in train]
    parents = sum(int(b['present'].sum()) for b, _ in training)
    absent = sum(len(b['present']) for b, _ in training) - parents
    weight = float(np.sqrt(parents / absent))
    trials = []
    for row in prior['trials']:
        folder = PRIOR['CACHE'] / ('profile-' + row['arm'])
        manifest = json.loads((folder / 'manifest.json').read_text())
        files = [folder / f'{i:02d}.npy' for i in range(len(training))]
        prepared = SimpleNamespace(samples=training, paths=files, bytes=row['bytes'])
        resident = Resident(prepared)
        # Rebuild every complete-group transform once before trusting the reuse.
        from research.focus_pair_appearance_head import blocks
        original = blocks(training, manifest['projection'], row['arm'])
        for expected, actual in zip(original, resident()):
            if any(not np.array_equal(expected[k], actual[k]) for k in expected):
                raise ValueError('Exact frozen transform and group replay required')
        times = []
        for _ in range(3):
            tick = time.monotonic()
            value, gradient = objective(np.zeros(manifest['dimension']), resident, weight)
            times.append(time.monotonic() - tick)
        if value != row['zero_objective'] or abs(np.linalg.norm(gradient) - row['gradient_norm']) > 1e-8:
            raise ValueError('Exact complete-fold objective and gradient norm required')
        trials.append(dict(arm=row['arm'], mode=resident.mode, objective_seconds=times,
                           zero_objective=value, gradient_norm=float(np.linalg.norm(gradient)),
                           exact_complete_transform_replay=True))
        print(json.dumps(trials[-1]), flush=True)
        del resident
    result = dict(status='verified_resident_full_fitting_fold_profile', trials=trials,
                  prepared_profile_sha256=sha(prior_path),
                  source_hashes={p: sha(ROOT / p) for p in ['research/focus_pair_appearance_resident.py',
                    'scripts/profile-focus-pair-appearance-resident.py',
                    'reports/experiments/focus-pair-appearance-resident-v1-design.md']},
                  held_out_not_evaluated=prior['held_out_not_evaluated'], training_stems=train,
                  quality_evaluated=False, authorized_for_submission=False, gpu_seconds=0,
                  elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT, timeout=300, check=True)
    else:
        raise ValueError('No arguments supported')
