"""One fixed fourteen-movie CPU uncertainty experiment on verified arrays."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_conditional_motion import predict as mean_predict, metrics as baseline_metrics
from research.focus_conditional_variance import fit, predict, metrics, gate

RUN = 'focus-conditional-variance-v1'
FILES = ['scripts/fit-focus-conditional-variance.py', 'research/focus_conditional_variance.py',
         f'reports/experiments/{RUN}-design.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    verified, final = runpy.run_path(str(ROOT / 'scripts/verify-focus-conditional-motion.py'))['verify']()
    prior = json.loads((ROOT / 'reports/experiments/focus-conditional-motion-v1-result.json').read_text())
    samples = {}
    for record in prior['records']:
        path = ROOT / '.biohub/cache/focus-conditional-motion-v1' / (record['stem'] + '.npz')
        if sha(path) != record['sha256']:
            raise ValueError('Actual verified fitting arrays changed')
        with np.load(path, allow_pickle=False) as data:
            samples[record['stem']] = {k: data[k].copy() for k in data.files}
    return verified, final, prior, samples


def main():
    started = time.monotonic()
    target = ROOT / f'reports/experiments/{RUN}-result.json'
    cache = ROOT / '.biohub/cache' / RUN
    if target.exists() or cache.exists():
        raise ValueError('Never overwrite completed or partial work')
    frozen = {p: sha(ROOT / p) for p in FILES}
    verified, original_final, previous, samples = load()
    stems = previous['training_stems']
    cache.mkdir()
    rows, folds = [], []
    for held, oldfold, oldrow in zip(stems, previous['folds'], previous['per_movie']):
        train = [s for s in stems if s != held]
        x = np.concatenate([samples[s]['x'] for s in train])
        y = np.concatenate([samples[s]['y'] for s in train])
        original = oldfold['model']
        model = fit(original, x, y, 'fitting')
        path = cache / (held + '-model.json')
        path.write_text(json.dumps(model, indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if restored != model:
            raise ValueError('Persisted parameters differ before held-out evaluation')
        sample = samples[held]
        mean = mean_predict(original, sample['x'])
        baseline = baseline_metrics(sample['y'], mean, original['variance_um2'])
        if baseline != oldrow['candidate'] or model['mean_model'] != original:
            raise ValueError('Exact old mean/control replay required')
        row = dict(stem=held, training_stems=train, baseline=baseline,
                   candidate=metrics(sample['y'], mean, predict(restored, sample['x'])))
        rows.append(row)
        folds.append(dict(held_out_stem=held, model_file=path.name, model_sha256=sha(path)))
        print(json.dumps(row), flush=True)
    decision = gate(rows)
    final = None
    if decision['passed']:
        x = np.concatenate([samples[s]['x'] for s in stems])
        y = np.concatenate([samples[s]['y'] for s in stems])
        model = fit(original_final, x, y, 'fitting')
        path = cache / 'model.json'
        path.write_text(json.dumps(model, indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if not np.array_equal(predict(model, x), predict(restored, x)):
            raise ValueError('Portable conditional variance replay failed')
        final = dict(file='model.json', sha256=sha(path), exact_prediction_replay=True)
    if any(sha(ROOT / p) != v for p, v in frozen.items()):
        raise ValueError('Frozen experiment changed during execution')
    result = dict(run_id=RUN, status='completed_conditional_variance_screen',
                  source_hashes=frozen, prior_verification=verified, training_stems=stems,
                  folds=folds, per_movie=rows, gate=decision, final_model=final,
                  gpu_seconds=0, source_movies_evaluated=0, new_target_movies_opened=0,
                  authorized_for_submission=False, elapsed_seconds=time.monotonic() - started,
                  caveat='Correction-only LOMO; not independent whole-system tracking validation')
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(gate=decision, final_model=final, elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       env=dict(os.environ), timeout=300, check=True)
    else:
        raise ValueError('Unsupported arguments')
