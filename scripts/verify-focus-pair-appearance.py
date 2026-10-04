"""Replay actual descriptors/moments and profile, without held-out scoring."""
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
from research.focus_candidate_ranker import pack, validate
from research.focus_pair_appearance import descriptors, empty_stats, accumulate, merge_stats, projector
from research.focus_pair_appearance_head import blocks, objective, metrics
from research.focus_balanced_candidate_ranker import frame_packet

BUILD = runpy.run_path(str(ROOT / 'scripts/build-focus-pair-appearance.py'))
RUN, CACHE, REPORT, sha = (BUILD[k] for k in ('RUN', 'CACHE', 'REPORT', 'sha'))


def read_stats(path):
    value = json.loads(path.read_text())
    return {k: np.asarray(v, dtype=np.int64 if k == 'count' else float) for k, v in value.items()}


def check_smoke(base, feature, model):
    expected_counts = dict(groups=len(base['starts']), parents=int(base['present'].sum()), choices=len(base['offset']))
    expected_counts['absent'] = expected_counts['groups'] - expected_counts['parents']
    weight = float(np.sqrt(expected_counts['parents'] / expected_counts['absent']))
    if model['counts'] != expected_counts or model['absent_weight'] != weight or model['parent_weight'] != 1.:
        raise ValueError('Exact smoke fitting counts and fixed class weight required')
    provider = lambda: blocks([(base, feature)], model['projection'], model['arm'])
    theta = np.asarray(model['theta'])
    value, gradient = objective(theta, provider, weight)
    initial, _ = objective(np.zeros(len(theta)), provider, weight)
    independent = .5 * np.dot(theta[1:], theta[1:])
    for block in provider():
        scores = block['offset'] + block['x'] @ theta
        weights = np.where(block['present'], 1., weight)
        for start, size, chosen, w in zip(block['starts'], block['sizes'], block['chosen'], weights):
            independent += w * (np.logaddexp.reduce(scores[start:start + size]) - scores[chosen])
    if (abs(value - model['objective']) > 1e-7 or abs(initial - model['initial_objective']) > 1e-7
            or abs(value - independent) > 1e-7 or not np.isfinite(gradient).all()):
        raise ValueError('Saved smoke objective/independent group loss replay differs')
    return dict(objective=value, independent_group_objective=float(independent),
                gradient_inf_norm=float(np.abs(gradient).max()), fitting_metrics=metrics([(base, feature)], model))


def load_receipt():
    path = REPORT / f'{RUN}-data-smoke.json'
    result = json.loads(path.read_text())
    if (result['status'] != 'verified_complete_fitting_appearance_data_and_real_smokes'
            or result['source_hashes'] != BUILD['sources']() or result['old_data_sha256'] != BUILD['OLD_DATA_SHA']
            or result['quality_evaluated'] is not False or result['authorized_for_submission'] is not False
            or any(result[k] != 0 for k in ('diagnostic_movies_opened', 'source_movies_opened',
                                          'new_target_movies_opened', 'gpu_seconds'))
            or result['total_groups'] != 10915 or result['total_choices'] != 4691320
            or result['descriptor_bytes'] != 1200977920 or result['real_pair_counts'] != [4669651, 10754]
            or [a['arm'] for a in result['stress']['arms']] != ['full', 'lda']):
        raise ValueError('Actual complete fixed-method fitting-only descriptor receipt required')
    return result, path


def verify():
    started = time.monotonic()
    result, path = load_receipt()
    evidence, parameters, originals = BUILD['inventory']()
    if [r['stem'] for r in result['records']] != [r['stem'] for r in evidence['records']]:
        raise ValueError('Exact twelve original fitting movies required')
    largest = min((-p['source_nodes'] * (p['known_parent'] + p['known_absent']), r['stem'], p['source_frame'])
                  for r, _, m in originals for p in m['pairs'] if p['known_parent'] and p['known_absent'])
    if (result['stress']['stem'], result['stress']['source_frame'], result['stress']['real_choices']) != (largest[1], largest[2], -largest[0]):
        raise ValueError('Declared real stress selection differs')
    checks = []
    for row, (old, folder, manifest) in zip(result['records'], originals):
        stem = row['stem']
        data_path = ROOT / '.biohub/cache/focus-candidate-ranker-v1' / (stem + '.npz')
        feature_path, moment_path = CACHE / (stem + '-appearance.npy'), CACHE / (stem + '-moments.json')
        if (sha(data_path) != row['base_sha256'] or row['base_sha256'] != old['sha256']
                or sha(feature_path) != row['descriptor_sha256'] or sha(moment_path) != row['moments_sha256']
                or row['manifest_sha256'] != old['manifest_sha256'] or row['raw_sha256'] != old['raw_sha256']
                or row['groups'] != old['groups'] or row['choices'] != old['choices']):
            raise ValueError('Actual descriptor/moment/base identities changed')
        base = BUILD['arrays'](data_path)
        validate(base)
        feature = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        if feature.shape != (old['choices'], 64) or feature.dtype != np.float32 or row['descriptor_bytes'] != feature.nbytes:
            raise ValueError('Complete actual descriptor array required')
        stats = empty_stats()
        cursor = 0
        for pair in manifest['pairs']:
            packet_path = folder / stem / pair['file']
            if sha(packet_path) != pair['sha256']:
                raise ValueError('Actual source feature packet changed')
            packet = BUILD['arrays'](packet_path)
            columns = np.flatnonzero(packet['labels'] >= 0)
            if not len(columns):
                continue
            packed = pack(packet, parameters, 'fitting')
            existing = frame_packet(base, int(packet['source_frame']))
            if any(not np.array_equal(packed[k], existing[k]) for k in packed):
                raise ValueError('Actual complete base/label identities differ')
            rebuilt = descriptors(packet, columns)
            actual = feature[cursor:cursor + len(rebuilt)]
            if not np.array_equal(rebuilt, actual):
                raise ValueError('Actual descriptors do not exactly rebuild from original image features')
            # Independent sufficient-statistic calculation, excluding null rows.
            real = np.ones(len(rebuilt), bool)
            real[packed['null_rows']] = False
            positive = np.zeros(len(rebuilt), bool)
            positive[packed['chosen'][packed['present'] == 1]] = True
            for cls, keep in [(0, real & ~positive), (1, real & positive)]:
                block = actual[keep].astype(float)
                stats['count'][cls] += len(block)
                stats['sums'][cls] += np.sum(block, axis=0)
                stats['seconds'][cls] += np.einsum('ni,nj->ij', block, block, optimize=True)
            cursor += len(rebuilt)
        saved = read_stats(moment_path)
        errors = {key: float(np.max(np.abs(stats[key] - saved[key]))) for key in stats}
        if (cursor != old['choices'] or saved['count'].tolist() != row['real_pair_counts']
                or any(not np.allclose(stats[k], saved[k], rtol=1e-10, atol=1e-8) for k in stats)):
            raise ValueError('Exact real-pair coverage or independent moment replay differs')
        checks.append(dict(stem=stem, descriptor_sha256=row['descriptor_sha256'], moment_max_errors=errors))
        del base, feature
    stem, frame = largest[1:]
    base = BUILD['arrays'](ROOT / '.biohub/cache/focus-candidate-ranker-v1' / (stem + '.npz'))
    packet = frame_packet(base, frame)
    selected = np.flatnonzero(base['source_frame'] == frame)
    begin = int(base['starts'][selected[0]])
    end = int(base['starts'][selected[-1]] + base['sizes'][selected[-1]])
    feature = np.load(CACHE / (stem + '-appearance.npy'), mmap_mode='r', allow_pickle=False)[begin:end].copy()
    stats = empty_stats()
    accumulate(stats, feature, packet, 'fitting')
    projection = projector(stats, 'fitting')
    if projection != result['stress']['projection']:
        raise ValueError('Actual smoke-only projection refit differs')
    smokes = []
    for arm in result['stress']['arms']:
        model_path = CACHE / (arm['arm'] + '-smoke-model.json')
        model = json.loads(model_path.read_text())
        if (sha(model_path) != arm['model_sha256'] or model['projection'] != projection
                or model['arm'] != arm['arm'] or len(model['theta']) != arm['dimensions']
                or model['iterations'] != arm['iterations'] or model['evaluations'] != arm['evaluations']
                or arm['exact_model_prediction_reload'] is not True or arm['gradient_relative_error'] > 2e-6):
            raise ValueError('Actual real smoke model, projection or execution metadata differs')
        smokes.append(dict(arm=arm['arm'], **check_smoke(packet, feature, model)))
    return dict(status='verified_complete_pair_appearance_replay', data_smoke_sha256=sha(path),
                verifier_sha256=sha(Path(__file__)), rows=checks, smoke=smokes,
                complete_descriptors_rebuilt=True, independent_moments_replayed=True,
                actual_smoke_projection_refit=True, smoke_optimizers_refit=False,
                complete_choices=4691320, complete_groups=10915, authorized_for_submission=False,
                quality_evaluated=False, gpu_seconds=0, elapsed_seconds=time.monotonic() - started)


def profile():
    started = time.monotonic()
    result, data_path = load_receipt()
    receipt_path = REPORT / f'{RUN}-verification.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'verified_complete_pair_appearance_replay'
            or receipt['data_smoke_sha256'] != sha(data_path) or receipt['verifier_sha256'] != sha(Path(__file__))):
        raise ValueError('Actual completed descriptor/smoke replay required')
    held = result['records'][0]['stem']
    records = [r for r in result['records'] if r['stem'] != held]
    samples, moments = [], []
    for row in records:
        stem = row['stem']
        base_path = ROOT / '.biohub/cache/focus-candidate-ranker-v1' / (stem + '.npz')
        app_path, stats_path = CACHE / (stem + '-appearance.npy'), CACHE / (stem + '-moments.json')
        if (sha(base_path) != row['base_sha256'] or sha(app_path) != row['descriptor_sha256']
                or sha(stats_path) != row['moments_sha256']):
            raise ValueError('Actual first-fold fitting data changed')
        base = BUILD['arrays'](base_path)
        validate(base)
        samples.append((base, np.load(app_path, allow_pickle=False, mmap_mode='r')))
        moments.append(read_stats(stats_path))
    projection = projector(merge_stats(moments, 'fitting'), 'fitting')
    parents = sum(int(b['present'].sum()) for b, _ in samples)
    absent = sum(int((b['present'] == 0).sum()) for b, _ in samples)
    weight = float(np.sqrt(parents / absent))
    trials = []
    for arm, dim in [('full', 72), ('lda', 9)]:
        timings = []
        for _ in range(2):
            tick = time.monotonic()
            value, gradient = objective(np.zeros(dim), lambda: blocks(samples, projection, arm), weight)
            timings.append(time.monotonic() - tick)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ValueError('Full-size objective must be finite')
        trials.append(dict(arm=arm, zero_objective=value, gradient_norm=float(np.linalg.norm(gradient)),
                           complete_fold_objective_seconds=timings))
    if abs(trials[0]['zero_objective'] - trials[1]['zero_objective']) > 1e-8:
        raise ValueError('Both arms must start from identical original physical logits')
    return dict(status='completed_first_fitting_fold_objective_profile', data_smoke_sha256=sha(data_path),
                verification_sha256=sha(receipt_path), profiler_sha256=sha(Path(__file__)),
                held_out_not_evaluated=held, training_stems=[r['stem'] for r in records],
                fitting_parents=parents, fitting_absent=absent,
                fitting_choices=sum(len(b['offset']) for b, _ in samples),
                first_fold_projection=projection, trials=trials, head_optimizer_run=False,
                quality_evaluated=False, authorized_for_submission=False, gpu_seconds=0,
                elapsed_seconds=time.monotonic() - started)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args in (['verify', '--worker'], ['profile', '--worker']):
        suffix = 'verification' if args[0] == 'verify' else 'profile'
        target = REPORT / f'{RUN}-{suffix}.json'
        if target.exists():
            raise ValueError('Never overwrite completed verification/profile')
        result = {'verify': verify, 'profile': profile}[args[0]]()
        target.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'first_fold_projection')}, indent=2), flush=True)
    elif args in (['verify'], ['profile']):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), args[0], '--worker'], cwd=ROOT,
                       env=dict(os.environ), timeout=900 if args[0] == 'verify' else 300, check=True)
    else:
        raise ValueError('Use verify or profile')
