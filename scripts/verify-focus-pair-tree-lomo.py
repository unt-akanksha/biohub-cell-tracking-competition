"""Independent native/tree/data/objective replay; never fits an optimizer."""
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
from research.focus_pair_appearance import projector, merge_stats
from research.focus_pair_appearance_head import blocks, validate_samples
from research.focus_pair_tree import portable_predict, SETTINGS, VERSION, ROUNDS

TRAIN = runpy.run_path(str(ROOT/'scripts/fit-focus-pair-tree-lomo.py'))
CACHE, REPORT, RUN = (TRAIN[k] for k in ('CACHE', 'REPORT', 'RUN'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


def native_model(path):
    import xgboost as xgb
    if xgb.__version__ != VERSION:
        raise ValueError('Original pinned native tree implementation required')
    return xgb.Booster(params=dict(nthread=2, device='cpu'), model_file=path)


def require_tree(model, counts, reused):
    if (model['settings'] != SETTINGS or model['version'] != VERSION or model['rounds'] != ROUNDS
            or model['role'] != 'fitting' or model['external_baseline_required'] is not True
            or model['authorized_for_submission'] is not False or len(model['trees']) != ROUNDS
            or model['training_groups'] != counts['groups'] or model['training_choices'] != counts['choices']
            or not np.isfinite(model['loss_trace']).all()
            or len(model['loss_trace']) != (25 if reused else 100)
            or not model['final_loss'] < model['initial_loss']):
        raise ValueError('Exact complete fixed fitting model and honest trace coverage required')
    if reused and (model['recovered_prefix_rounds'] != 75 or model['loss_trace_start_round'] != 76):
        raise ValueError('Recovered prefix trace must be identified honestly')
    # Traverse every node even with no prediction examples; checks depth and leaf bounds.
    portable_predict(model, np.empty((0, 72), np.float32))


def independent_group_metrics(score, block):
    if np.asarray(score).shape != block['offset'].shape or not np.isfinite(score).all():
        raise ValueError('Finite scores for every complete real/null group required')
    result = dict(loss_sum=0., known_parent=0, known_absent=0, correct_parent=0, correct_absent=0)
    for start, size, chosen, parent in zip(block['starts'], block['sizes'], block['chosen'], block['present']):
        group = score[start:start+size]
        result['loss_sum'] += float(np.logaddexp.reduce(group)-score[chosen])
        key = 'parent' if parent else 'absent'
        result['known_'+key] += 1
        result['correct_'+key] += int(start+int(np.argmax(group)) == chosen)
    return result


def weighted_loss(score, block, absent_weight):
    # Deliberately avoid the fitting derivatives/objective callback.
    total = 0.
    for start, size, chosen, parent in zip(block['starts'], block['sizes'], block['chosen'], block['present']):
        normalizer = np.logaddexp.reduce(score[start:start+size])
        total += (1. if parent else absent_weight)*float(normalizer-score[chosen])
    return total


def verify_fold(row, execution, samples, moments, controls, profile):
    held, reused = row['held_out'], row['reused_profile']
    train = [s for s in samples if s != held]
    folder, model_path = CACHE/held, CACHE/held/'model.json'
    composite = json.loads(model_path.read_text())
    if (sha(model_path) != row['model_sha256'] or composite['held_out'] != held
            or composite['training_stems'] != train or row['training_stems'] != train
            or composite['reused_profile'] != reused or composite['authorized_for_submission'] is not False
            or execution['held_out'] != held or execution['actual_worker_pid_confirmed'] is not True
            or execution['peak_working_bytes'] > TRAIN['MAX_BYTES']
            or execution != json.loads((folder/'execution.json').read_text())
            or row != json.loads((folder/'result.json').read_text())):
        raise ValueError('Exact persisted model, fold identity, result and actual-worker resource receipt required')
    baseline_path = TRAIN['LINEAR']['CACHE']/'full'/f'{held}-model.json'
    baseline = json.loads(baseline_path.read_text())['model']
    if composite['baseline'] != baseline or composite['baseline_sha256'] != sha(baseline_path):
        raise ValueError('Exact original fold-only baseline required')
    fitting = [samples[s] for s in train]
    counts = validate_samples(fitting)
    projection = projector(merge_stats([moments[s] for s in train], 'fitting'), 'fitting')
    if (counts != baseline['counts'] or projection != baseline['projection']
            or baseline['absent_weight'] != np.sqrt(counts['parents']/counts['absent'])):
        raise ValueError('Original eleven-movie counts, projection and class weight required')
    tree = composite['tree']
    require_tree(tree, counts, reused)
    if reused:
        fitting_folder = TRAIN['RECOVERY']['OLD']['CACHE']
        fit_folder = TRAIN['RECOVERY']['CACHE']/'fit'
        if sha(fit_folder/'portable.json') != profile['worker']['tree_sha256']:
            raise ValueError('Verified recovered first-fold artifact required')
    else:
        fitting_folder, fit_folder = folder, folder/'fit'
    if tree != json.loads((fit_folder/'portable.json').read_text()):
        raise ValueError('Composite must contain the exact separately persisted tree model')
    native = native_model(fit_folder/'native.ubj')
    if [json.loads(t) for t in native.get_dump(dump_format='json')] != tree['trees']:
        raise ValueError('Native and portable100 trees differ')
    checkpoints = [25, 50, 75] if reused else [25, 50, 75, 100]
    prefix_75 = None
    for n in checkpoints:
        checkpoint = native_model(fitting_folder/'fit'/f'partial-{n:03d}.ubj')
        if [json.loads(t) for t in checkpoint.get_dump(dump_format='json')] != tree['trees'][:n]:
            raise ValueError('Every saved checkpoint must be the actual final-model prefix')
        if reused and n == 75:
            prefix_75 = checkpoint
    prepared = np.load(fitting_folder/'features.npy', mmap_mode='r', allow_pickle=False)
    margins = np.load(fitting_folder/'margin.npy', mmap_mode='r', allow_pickle=False)
    with np.load(fitting_folder/'groups.npz', allow_pickle=False) as saved:
        groups = {k: saved[k].copy() for k in saved.files}
    if prepared.shape != (counts['choices'], 72) or prepared.dtype != np.float32 or margins.dtype != np.float64:
        raise ValueError('Complete original prepared data types/shapes required')
    position, group_position, initial, final, prefix_loss = 0, 0, 0., 0., 0.
    for block in blocks(fitting, projection, 'full'):
        end, group_end = position+len(block['offset']), group_position+len(block['starts'])
        features = block['x'].astype(np.float32)
        margin = block['offset']+block['x'] @ np.asarray(baseline['theta'])
        if not np.array_equal(prepared[position:end], features) or not np.array_equal(margins[position:end], margin):
            raise ValueError('Prepared features/baseline must exactly replay from original11 fitting movies')
        for key in ('starts', 'sizes', 'chosen', 'present'):
            expected = block[key]+position if key in ('starts', 'chosen') else block[key]
            if not np.array_equal(groups[key][group_position:group_end], expected):
                raise ValueError('Every fitting target/candidate/label must be accounted for')
        prediction = native.inplace_predict(features, predict_type='margin')
        initial += weighted_loss(margin, block, baseline['absent_weight'])
        final += weighted_loss(margin+prediction, block, baseline['absent_weight'])
        if prefix_75 is not None:
            prefix_loss += weighted_loss(margin+prefix_75.inplace_predict(features, predict_type='margin'), block, baseline['absent_weight'])
        position, group_position = end, group_end
    if position != counts['choices'] or group_position != counts['groups']:
        raise ValueError('Complete training coverage required')
    penalty = .5*np.dot(baseline['theta'][1:], baseline['theta'][1:])
    if (abs(initial-tree['initial_loss']) > 1e-6 or abs(final-tree['final_loss']) > 1e-6
            or abs(initial+penalty-baseline['objective']) > 1e-6
            or (reused and abs(prefix_loss-tree['prefix_loss']) > 1e-6)):
        raise ValueError('Native independent full fitting likelihoods differ')
    held_result = dict(loss_sum=0., known_parent=0, known_absent=0, correct_parent=0, correct_absent=0)
    max_error = 0.
    # Different block size and native predictions; independent group logaddexp/argmax.
    for block in blocks([samples[held]], projection, 'full', group_block=17):
        native_values = native.inplace_predict(block['x'].astype(np.float32), predict_type='margin')
        portable_values = portable_predict(tree, block['x'])
        max_error = max(max_error, float(np.max(np.abs(native_values-portable_values))))
        np.testing.assert_allclose(native_values, portable_values, rtol=0, atol=2e-6)
        margin = block['offset']+block['x'] @ np.asarray(baseline['theta'])
        part = independent_group_metrics(margin+native_values, block)
        for key in held_result:
            held_result[key] += part[key]
    held_result['nll'] = held_result['loss_sum']/(held_result['known_parent']+held_result['known_absent'])
    if any(abs(held_result[k]-row['candidate'][k]) > 1e-6 for k in held_result):
        raise ValueError('Held-out native decisions and likelihood failed replay')
    for key in ('physical', 'neural', 'weighted_ranker', 'lda'):
        if row[key] != controls[held][key]:
            raise ValueError('Original comparison controls changed')
    if row['full_linear'] != controls[held]['candidate']:
        raise ValueError('Full linear reference changed')
    return dict(held_out=held, model_sha256=sha(model_path), complete_training_choices=position,
        complete_training_groups=group_position, initial_loss=initial, final_loss=final,
        prefix_loss=prefix_loss if reused else None, checked_checkpoints=checkpoints,
        held_out_native_portable_max_error=max_error, independent_held_out=held_result)


def main():
    started = time.monotonic()
    output = REPORT/f'{RUN}-verification.json'
    if output.exists():
        raise ValueError('Never overwrite completed independent verification')
    result_path = REPORT/f'{RUN}-result.json'
    result = json.loads(result_path.read_text())
    profile, controls = TRAIN['evidence']()
    if (result['status'] != 'completed' or result['failure'] is not None or len(result['folds']) != 12
            or result['final_model'] is not None or result['source_hashes'] != TRAIN['sources']()
            or json.loads((CACHE/'launch.json').read_text())['source_hashes'] != TRAIN['sources']()):
        raise ValueError('Complete terminal unchanged twelve-fold screen required')
    _, samples, moments, _, _ = TRAIN['LINEAR']['load']()
    if ([r['held_out'] for r in result['folds']] != list(samples)
            or [r['held_out'] for r in result['executions']] != list(samples)):
        raise ValueError('Exact twelve-fold model/resource order required')
    checks = []
    for row, execution in zip(result['folds'], result['executions']):
        checks.append(verify_fold(row, execution, samples, moments, controls, profile))
        print(json.dumps(checks[-1]), flush=True)
    decision = TRAIN['LINEAR']['BASE']['gate'](result['folds'])
    if decision != result['gate'] or result['source_hashes'] != TRAIN['sources']():
        raise ValueError('Unchanged original acceptance must replay')
    record = dict(status='independently_verified_complete_nonlinear_candidate_lomo', result_sha256=sha(result_path),
        verifier_sha256=sha(Path(__file__)), source_hashes=result['source_hashes'], checks=checks,
        gate=decision, optimizer_refit=False, diagnostic_movies_opened=0, source_movies_opened=0,
        new_target_movies_opened=0, gpu_seconds=0, authorized_for_submission=False,
        elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps(dict(gate=decision, elapsed_seconds=record['elapsed_seconds']), indent=2), flush=True)


def smoke():
    """One already completed held-out fold; no complete-run verification claim."""
    started = time.monotonic()
    output = REPORT/f'{RUN}-verifier-smoke.json'
    if output.exists():
        raise ValueError('Never overwrite completed verifier smoke')
    profile, controls = TRAIN['evidence']()
    held = next(iter(controls))
    model_path, result_path = CACHE/held/'model.json', CACHE/held/'result.json'
    composite, row = [json.loads(p.read_text()) for p in (model_path, result_path)]
    if (sha(model_path) != row['model_sha256'] or composite['held_out'] != held
            or composite['training_stems'] != [s for s in controls if s != held]):
        raise ValueError('Actual first completed fold-only composite required')
    baseline, tree = composite['baseline'], composite['tree']
    require_tree(tree, baseline['counts'], True)
    native_path = TRAIN['RECOVERY']['CACHE']/'fit/native.ubj'
    if sha(native_path) != profile['worker']['native_sha256']:
        raise ValueError('Original first-fold native model required')
    native = native_model(native_path)
    if [json.loads(t) for t in native.get_dump(dump_format='json')] != tree['trees']:
        raise ValueError('Actual native/portable100-tree structure differs')
    receipt_path = REPORT/'focus-pair-appearance-v1-data-smoke.json'
    if sha(receipt_path) != '4e2eb70fa910025184a53cc2549293658496ad77803bc403a82137254af9c22f':
        raise ValueError('Original known-target array receipt required')
    source = next(r for r in json.loads(receipt_path.read_text())['records'] if r['stem'] == held)
    base_path = ROOT/'.biohub/cache/focus-candidate-ranker-v1'/f'{held}.npz'
    app_path = ROOT/'.biohub/cache/focus-pair-appearance-v1'/f'{held}-appearance.npy'
    if sha(base_path) != source['base_sha256'] or sha(app_path) != source['descriptor_sha256']:
        raise ValueError('Exact original held-out candidate arrays required')
    with np.load(base_path, allow_pickle=False) as data:
        base = {k:data[k].copy() for k in data.files}
    app = np.load(app_path, mmap_mode='r', allow_pickle=False)
    actual = dict(loss_sum=0., known_parent=0, known_absent=0, correct_parent=0, correct_absent=0)
    rows, error = 0, 0.
    for block in blocks([(base,app)],baseline['projection'],'full',group_block=17):
        values = native.inplace_predict(block['x'].astype(np.float32),predict_type='margin')
        reference = portable_predict(tree,block['x'])
        error = max(error,float(np.max(np.abs(values-reference))))
        score = block['offset']+block['x'] @ np.asarray(baseline['theta'])+values
        part = independent_group_metrics(score,block)
        for key in actual:
            actual[key] += part[key]
        rows += len(values)
    actual['nll'] = actual['loss_sum']/(actual['known_parent']+actual['known_absent'])
    if (rows != len(base['offset']) or error > 2e-6
            or any(abs(actual[k]-row['candidate'][k]) > 1e-6 for k in actual)):
        raise ValueError('Complete first-fold independent native/portable/held-out replay failed')
    result = dict(status='completed_first_fold_native_verifier_smoke',held_out=held,
        model_sha256=sha(model_path),result_sha256=sha(result_path),source_sha256=sha(Path(__file__)),
        complete_choices=rows,independent_metrics=actual,native_portable_max_error=error,
        full_twelve_fold_verification_completed=False,training_objective_replayed=False,optimizer_run=False,
        diagnostic_movies_opened=0,source_movies_opened=0,new_target_movies_opened=0,
        gpu_seconds=0,authorized_for_submission=False,elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       timeout=1800, check=True)
    elif sys.argv[1:] == ['--smoke']:
        smoke()
    else:
        raise ValueError('No arguments supported')
