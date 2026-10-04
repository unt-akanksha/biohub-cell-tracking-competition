"""Replay completed LOMO arms against actual fold-only projections and all groups."""
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
from research.focus_pair_appearance import merge_stats, projector
from research.focus_pair_appearance_head import blocks, objective, metrics, validate_samples

TRAIN = runpy.run_path(str(ROOT / 'scripts/fit-focus-pair-appearance-resident.py'))
CACHE, REPORT, RUN, sha = (TRAIN[k] for k in ('CACHE', 'REPORT', 'RUN', 'sha'))


def verify(arm):
    started = time.monotonic()
    target = REPORT / f'{RUN}-{arm}-verification.json'
    if target.exists():
        raise ValueError('Never overwrite completed arm verification')
    launch = json.loads((CACHE / 'launch.json').read_text())
    if launch['source_hashes'] != TRAIN['sources']():
        raise ValueError('Actual launched method changed')
    result_path = CACHE / arm / 'result.json'
    result = json.loads(result_path.read_text())
    if result['arm'] != arm or result['status'] != 'completed' or len(result['folds']) != 12:
        raise ValueError('All twelve completed fits required; no partial quality claim')
    evidence, samples, moments, controls, weighted = TRAIN['load']()
    stems = list(samples)
    if [r['held_out'] for r in result['folds']] != stems or result['final_model'] is not None:
        raise ValueError('Exact twelve-movie evaluation and no premature final fit required')
    checks, folds = [], []
    for row in result['folds']:
        held = row['held_out']
        train = [s for s in stems if s != held]
        model_path = CACHE / arm / (held + '-model.json')
        row_path = CACHE / arm / (held + '-result.json')
        saved = json.loads(model_path.read_text())
        model = saved['model']
        projection = projector(merge_stats([moments[s] for s in train], 'fitting'), 'fitting')
        fitting = [samples[s] for s in train]
        counts = validate_samples(fitting)
        weight = float(np.sqrt(counts['parents'] / counts['absent']))
        if (sha(model_path) != row['model_sha256'] or saved['training_stems'] != train
                or saved['held_out'] != held or row['training_stems'] != train
                or model['projection'] != projection or model['counts'] != counts
                or model['absent_weight'] != weight or model['parent_weight'] != 1.
                or model['arm'] != arm or json.loads(row_path.read_text()) != row):
            raise ValueError('Actual fold-only projection/scaler/weights/identity changed')
        theta = np.asarray(model['theta'])
        provider = lambda: blocks(fitting, projection, arm)
        value, gradient = objective(theta, provider, weight)
        if abs(value - model['objective']) > 1e-7:
            raise ValueError('Actual training objective does not replay through original stream')
        # Independently evaluate each held-out complete group, not the trainer's reduction.
        actual = metrics([samples[held]], model)
        independent = dict(loss_sum=0., known_parent=0, known_absent=0, correct_parent=0, correct_absent=0)
        decisions = []
        for block in blocks([samples[held]], projection, arm):
            scores = block['offset'] + block['x'] @ theta
            for start, size, chosen, parent in zip(block['starts'], block['sizes'], block['chosen'], block['present']):
                group = scores[start:start+size]
                selected = int(np.argmax(group)) + start
                independent['loss_sum'] += float(np.logaddexp.reduce(group) - scores[chosen])
                label = 'parent' if parent else 'absent'
                independent['known_' + label] += 1
                independent['correct_' + label] += int(selected == chosen)
                decisions.append(selected-int(start))
        independent['nll'] = independent['loss_sum'] / len(decisions)
        if any(actual[k] != row['candidate'][k] for k in actual):
            raise ValueError('Every stored held-out metric must replay exactly')
        if any(abs(independent[k] - actual[k]) > 1e-7 for k in independent):
            raise ValueError('Independent per-group held-out decisions/loss differ')
        if (row['physical'] != controls[held]['physical'] or row['neural'] != controls[held]['neural']
                or row['weighted_ranker'] != weighted[held]['candidate']):
            raise ValueError('Unchanged original controls required')
        checks.append(dict(held_out=held, model_sha256=sha(model_path), training_objective=value,
                           gradient_inf_norm=float(np.abs(gradient).max()),
                           projection_refit_exact=True, independent_held_out=independent))
        folds.append(row)
        print(json.dumps(dict(arm=arm, verified=held, candidate=actual)), flush=True)
    decision = TRAIN['BASE']['gate'](folds)
    if decision != result['gate'] or TRAIN['sources']() != launch['source_hashes']:
        raise ValueError('Original acceptance or frozen method changed')
    verified = dict(status='host_verified_complete_pair_appearance_lomo_arm', arm=arm,
                    result_sha256=sha(result_path), launch_source_hashes=launch['source_hashes'],
                    verifier_sha256=sha(Path(__file__)), checks=checks, gate=decision,
                    optimizer_refit=False, diagnostic_movies_opened=0, source_movies_opened=0,
                    new_target_movies_opened=0, gpu_seconds=0, authorized_for_submission=False,
                    elapsed_seconds=time.monotonic()-started)
    target.write_text(json.dumps(verified, indent=2, allow_nan=False))
    print(json.dumps(dict(arm=arm, gate=decision, elapsed_seconds=verified['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2 and args[0] in ('lda', 'full') and args[1] == '--worker':
        verify(args[0])
    elif len(args) == 1 and args[0] in ('lda', 'full'):
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), args[0], '--worker'],
                       cwd=ROOT, timeout=900, check=True)
    else:
        raise ValueError('Choose one completed arm: lda or full')
