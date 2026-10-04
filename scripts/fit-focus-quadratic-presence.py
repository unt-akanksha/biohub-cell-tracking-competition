"""Bounded CPU-only LOMO screen; never opens diagnostic or target summaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_parent_presence import fit as linear_fit, metrics
from research.focus_quadratic_presence import fit, evaluate, screen
from research.focus_summary_validation import validate_summary

RUN = 'focus-quadratic-presence-v1'
PRIOR_SHA = 'ad1a546a0c903358c37ab20aee206cd0c617b093a1aa122254721f178ca70dd0'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fitting():
    prior = ROOT / 'reports/experiments/focus-expanded-presence-v1-result.json'
    if sha(prior) != PRIOR_SHA:
        raise ValueError('Frozen upstream verification receipt required')
    receipt = json.loads(prior.read_text())
    for relative, digest in receipt['source_hashes'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Upstream verified implementation changed')
    evidence = receipt['evidence']
    contract = evidence['contract']
    records = evidence['records']
    stems = contract['fitting_stems']
    if (len(stems) != 12 or len(set(stems)) != 12
            or [(r['stem'], r['role']) for r in records] != [(s, 'fitting') for s in stems]
            or set(stems) & set(contract['unchanged_diagnostic_stems'])):
        raise ValueError('Exact disjoint fitting scope required')
    folder = ROOT / '.biohub/cache/kernel-outputs/focus-expanded-summary-v1/focus_expanded_summary/outputs'
    if sha(folder / 'result.json') != evidence['worker_result_sha256']:
        raise ValueError('Actual verified summary worker result changed')
    actual = json.loads((folder / 'result.json').read_text())
    if len(actual['records']) != 12:
        raise ValueError('Twelve actual fitting records required')
    movies = {}
    for recorded, worker in zip(records, actual['records']):
        if any(recorded[k] != worker[k] for k in worker):
            raise ValueError('Actual summary record differs from prior verification')
        stem = recorded['stem']
        path = folder / (stem + '.npz')
        label_dir = 'focus-adaptation-labels-v1' if stem in contract['previous_fitting_stems'] else 'focus-extra-fit-labels-v1'
        labels_path = ROOT / '.biohub/cache' / label_dir / (stem + '.json')
        if sha(path) != recorded['sha256'] or sha(labels_path) != recorded['labels_sha256']:
            raise ValueError('Actual fitting summary or label bytes changed')
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        validate_summary(data, recorded, json.loads(labels_path.read_text()))
        movies[stem] = data
    if sum(len(d['present']) for d in movies.values()) != 10915 or sum(int(d['present'].sum()) for d in movies.values()) != 10754:
        raise ValueError('Exact 10754 parent and161 absent labels required')
    return movies, dict(prior_receipt_sha256=sha(prior), records=records,
                        worker_result_sha256=evidence['worker_result_sha256'],
                        diagnostic_arrays_opened=False, verification_scope='Frozen actual summary hashes and full fitting-label identity; no new encoder replay')


def concatenate(movies, stems):
    return {key: np.concatenate([movies[stem][key] for stem in stems]) for key in movies[stems[0]]}


def main():
    started = time.monotonic()
    target = ROOT / f'reports/experiments/{RUN}-result.json'
    cache = ROOT / '.biohub/cache' / RUN
    if target.exists() or cache.exists():
        raise ValueError('Never overwrite a fitted or partially evaluated experiment')
    sources = {p: sha(ROOT / p) for p in (
        'research/focus_quadratic_presence.py', 'scripts/fit-focus-quadratic-presence.py',
        'reports/experiments/focus-quadratic-presence-v1-design.md',
        'research/focus_parent_presence.py', 'research/focus_summary_validation.py')}
    movies, evidence = load_fitting()
    cache.mkdir()
    folds = []
    for held_out in movies:
        fitting_stems = [stem for stem in movies if stem != held_out]
        data = concatenate(movies, fitting_stems)
        linear = linear_fit(data, 'fitting')
        quadratic = fit(data, 'fitting')
        saved = dict(held_out=held_out, fitting_stems=fitting_stems, linear=linear, quadratic=quadratic)
        path = cache / (held_out + '-models.json')
        path.write_text(json.dumps(saved, indent=2, allow_nan=False))
        recovered = json.loads(path.read_text())
        if recovered != saved:
            raise ValueError('Saved coefficients must replay exactly before held-out evaluation')
        held = movies[held_out]
        fold = dict(held_out=held_out, fitting_stems=fitting_stems, models_sha256=sha(path),
                    original=metrics(held), linear=metrics(held, recovered['linear']),
                    quadratic=evaluate(held, recovered['quadratic']))
        folds.append(fold)
        (cache / (held_out + '-result.json')).write_text(json.dumps(fold, indent=2, allow_nan=False))
        print(json.dumps(dict(held_out=held_out, nll={arm: fold[arm]['nll'] for arm in ('original', 'linear', 'quadratic')})), flush=True)
    gate = screen(folds)
    if any(sha(ROOT / p) != digest for p, digest in sources.items()):
        raise ValueError('Frozen method changed during fitting')
    result = dict(run_id=RUN, status='completed_twelve_movie_quadratic_lomo', evidence=evidence,
                  source_hashes=sources, folds=folds, screening_gate=gate,
                  gpu_seconds=0, elapsed_seconds=time.monotonic() - started,
                  diagnostic_evaluated=False, source_selection_opened=False,
                  new_target_movies_opened=0, final_model_fitted=False, authorized_for_submission=False)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ('status', 'screening_gate', 'elapsed_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                       cwd=ROOT, env=dict(os.environ), timeout=300, check=True)
    else:
        raise ValueError('Only the bounded default launcher or internal worker is supported')
