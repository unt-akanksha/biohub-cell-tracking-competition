"""One fixed correlated-motion screen on previously verified training arrays."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_conditional_motion import predict, metrics as diagonal_metrics
from research.focus_correlated_motion import covariance, metrics, gate
RUN = 'focus-correlated-motion-v1'
FILES = ['scripts/fit-focus-correlated-motion.py', 'research/focus_correlated_motion.py',
         'reports/experiments/focus-correlated-motion-v1-design.md']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    target = ROOT / f'reports/experiments/{RUN}-result.json'
    cache = ROOT / '.biohub/cache' / RUN
    if target.exists() or cache.exists(): raise ValueError('Never overwrite a completed or partial experiment')
    frozen = {f: sha(ROOT/f) for f in FILES}
    prior_receipt, original_final = runpy.run_path(str(ROOT/'scripts/verify-focus-conditional-motion.py'))['verify']()
    previous = json.loads((ROOT/'reports/experiments/focus-conditional-motion-v1-result.json').read_text())
    stems = previous['training_stems']; samples = {}
    for row in previous['records']:
        path = ROOT/'.biohub/cache/focus-conditional-motion-v1'/(row['stem']+'.npz')
        if sha(path) != row['sha256']: raise ValueError('Verified fitting arrays changed')
        with np.load(path, allow_pickle=False) as data: samples[row['stem']] = {k: data[k].copy() for k in data.files}
    folds = []; rows = []
    for held, oldfold, oldrow in zip(stems, previous['folds'], previous['per_movie']):
        train = [s for s in stems if s != held]
        x = np.concatenate([samples[s]['x'] for s in train]); y = np.concatenate([samples[s]['y'] for s in train])
        model = oldfold['model']; cov = covariance(y - predict(model, x))
        sample = samples[held]; mean = predict(model, sample['x'])
        base = diagonal_metrics(sample['y'], mean, model['variance_um2'])
        if base != oldrow['candidate']: raise ValueError('Exact original held-out mean/diagonal evaluation required')
        row = dict(stem=held, training_stems=train, baseline=base, candidate=metrics(sample['y'], mean, cov))
        rows.append(row); folds.append(dict(held_out_stem=held, covariance_um2=cov.tolist()))
        print(json.dumps(dict(stage='heldout_covariance', **row)), flush=True)
    decision = gate(rows); final = None
    if decision['passed']:
        x = np.concatenate([samples[s]['x'] for s in stems]); y = np.concatenate([samples[s]['y'] for s in stems])
        cov = covariance(y - predict(original_final, x))
        model = dict(mean_model=original_final, covariance_um2=cov.tolist(), training_stems=stems)
        cache.mkdir(); path = cache/'model.json'; path.write_text(json.dumps(model, indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if metrics(y, predict(original_final, x), cov) != metrics(y, predict(restored['mean_model'], x), restored['covariance_um2']):
            raise ValueError('Portable density replay failed')
        final = dict(file='model.json', sha256=sha(path), exact_density_replay=True)
    if frozen != {f: sha(ROOT/f) for f in FILES}: raise ValueError('Frozen design/code changed')
    result = dict(status='completed_correlated_motion_screen', run_id=RUN, source_hashes=frozen,
                  prior_verification=prior_receipt, training_stems=stems, folds=folds, per_movie=rows,
                  gate=decision, final_model=final, gpu_seconds=0, source_movies_evaluated=0,
                  new_target_movies_opened=0, authorized_for_submission=False, elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(gate=decision, final_model=final)), flush=True)


if __name__ == '__main__': main()
