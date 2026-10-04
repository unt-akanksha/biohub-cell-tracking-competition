"""Fixed full-candidate LOMO ranking, after verified real-data functionality."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_candidate_ranker import combine, fit, metrics, validate

RUN = 'focus-candidate-ranker-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    path = ROOT / f'reports/experiments/{RUN}-data-smoke.json'
    evidence = json.loads(path.read_text())
    if (evidence['status'] != 'verified_fitting_candidate_data_and_real_smoke'
            or len(evidence['records']) != 12 or evidence['total_groups'] != 10915
            or evidence['stress_smoke']['exact_parameter_roundtrip'] is not True
            or evidence['stress_smoke']['quality_evaluated'] is not False):
        raise ValueError('Complete actual full-candidate data and real smoke required')
    if any(sha(ROOT / p) != value for p, value in evidence['source_hashes'].items()):
        raise ValueError('Frozen candidate data/ranking method changed')
    cache = ROOT / '.biohub/cache' / RUN
    if sha(cache / 'stress-model.json') != evidence['stress_smoke']['model_sha256']:
        raise ValueError('Actual persisted real smoke model changed')
    samples = {}
    for row in evidence['records']:
        file = cache / (row['stem'] + '.npz')
        if sha(file) != row['sha256']:
            raise ValueError('Actual candidate choice arrays changed')
        with np.load(file, allow_pickle=False) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        validate(data)
        if (len(data['starts']) != row['groups'] or len(data['offset']) != row['choices']
                or sum(a.nbytes for a in data.values()) != row['bytes'] or metrics(data) != row['physical']):
            raise ValueError('Exact original physical posterior and candidate coverage required')
        samples[row['stem']] = data
    if len(samples) != 12 or sum(len(s['offset']) for s in samples.values()) != evidence['total_choices']:
        raise ValueError('Complete unique candidate inventory required')
    return evidence, samples


def pooled(rows):
    keys = ('loss_sum', 'known_parent', 'known_absent', 'correct_parent', 'correct_absent')
    result = {key: sum(r[key] for r in rows) for key in keys}
    result['nll'] = result['loss_sum'] / (result['known_parent'] + result['known_absent'])
    return result


def gate(folds):
    if len(folds) != 12 or len({r['held_out'] for r in folds}) != 12:
        raise ValueError('Twelve unique movie-held-out fits required')
    for row in folds:
        if len({(row[a]['known_parent'], row[a]['known_absent']) for a in ('physical', 'neural', 'candidate')}) != 1:
            raise ValueError('Identical original labels across all arms required')
    totals = {a: pooled([r[a] for r in folds]) for a in ('physical', 'neural', 'candidate')}
    new, controls = totals['candidate'], [totals['physical'], totals['neural']]
    conditions = dict(nll_below_both=new['nll'] < min(c['nll'] for c in controls),
                      parent_not_lower=new['correct_parent'] >= max(c['correct_parent'] for c in controls),
                      absent_not_lower=new['correct_absent'] >= max(c['correct_absent'] for c in controls),
                      eight_movie_nll_wins=sum(r['candidate']['nll'] < r['neural']['nll'] for r in folds) >= 8,
                      per_movie_nll_loss_bounded=max(r['candidate']['nll'] - r['neural']['nll'] for r in folds) <= .02)
    return dict(pooled=totals, conditions=conditions, passed=all(conditions.values()),
                scope='New ranking correction LOMO only; original encoder used fitting movies')


def main():
    started = time.monotonic()
    target = ROOT / f'reports/experiments/{RUN}-lomo.json'
    cache = ROOT / '.biohub/cache' / RUN / 'lomo'
    if target.exists() or cache.exists():
        raise ValueError('Never overwrite complete or partial ranking experiment')
    sources = {p: sha(ROOT / p) for p in ('scripts/fit-focus-candidate-ranker.py',
               'research/focus_candidate_ranker.py', f'reports/experiments/{RUN}-design.md')}
    evidence, samples = load()
    cache.mkdir()
    records = {r['stem']: r for r in evidence['records']}
    folds = []
    for held_out in samples:
        fold_started = time.monotonic()
        train = [stem for stem in samples if stem != held_out]
        fitting = combine([samples[stem] for stem in train])
        model = fit(fitting, 'fitting')
        path = cache / (held_out + '-model.json')
        path.write_text(json.dumps(dict(training_stems=train, held_out=held_out, model=model), indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if restored['model'] != model:
            raise ValueError('Actual saved model changed before held-out evaluation')
        del fitting
        row = dict(held_out=held_out, training_stems=train, model_sha256=sha(path),
                   physical=records[held_out]['physical'], neural=records[held_out]['neural'],
                   candidate=metrics(samples[held_out], restored['model']), elapsed_seconds=time.monotonic() - fold_started)
        folds.append(row)
        (cache / (held_out + '-result.json')).write_text(json.dumps(row, indent=2, allow_nan=False))
        print(json.dumps(dict(held_out=held_out, candidate=row['candidate'], elapsed_seconds=row['elapsed_seconds'])), flush=True)
    decision = gate(folds)
    final = None
    if decision['passed']:
        model = fit(combine(list(samples.values())), 'fitting')
        path = cache / 'model.json'
        path.write_text(json.dumps(model, indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        if any(metrics(s, restored) != metrics(s, model) for s in samples.values()):
            raise ValueError('Final portable posterior/decision replay differs')
        final = dict(file='model.json', sha256=sha(path), exact_prediction_replay=True)
    if any(sha(ROOT / p) != value for p, value in sources.items()):
        raise ValueError('Fixed candidate method changed during evaluation')
    result = dict(status='completed_full_candidate_ranker_lomo', run_id=RUN, source_hashes=sources,
                  data_smoke_sha256=sha(ROOT / f'reports/experiments/{RUN}-data-smoke.json'),
                  folds=folds, gate=decision, final_model=final,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  gpu_seconds=0, authorized_for_submission=False, elapsed_seconds=time.monotonic() - started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(gate=decision, final_model=final, elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       env=dict(os.environ), timeout=900, check=True)
    else:
        raise ValueError('Unsupported arguments')
