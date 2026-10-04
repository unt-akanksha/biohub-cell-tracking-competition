"""Complete fitting appearance export, exact geometry replay and real two-arm smoke."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_candidate_ranker import pack, validate
from research.focus_cached_pair import validate_pair
from research.focus_balanced_candidate_ranker import frame_packet
from research.focus_pair_appearance import descriptors, empty_stats, accumulate, projector
from research.focus_pair_appearance_head import fit, metrics, objective, blocks

RUN = 'focus-pair-appearance-v1'
CACHE = ROOT / '.biohub/cache' / RUN
REPORT = ROOT / 'reports/experiments'
OLD_DATA = REPORT / 'focus-candidate-ranker-v1-data-smoke.json'
OLD_DATA_SHA = 'c0376562166ca50ffdb12e9f79aff7bc867a217c5f69104454e723e45c3b8ccf'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def sources():
    return {p: sha(ROOT / p) for p in (
        'scripts/build-focus-pair-appearance.py', 'research/focus_pair_appearance.py',
        'research/focus_pair_appearance_head.py', 'research/focus_candidate_ranker.py',
        'research/focus_balanced_candidate_ranker.py', 'research/focus_cached_pair.py',
        'tests/test_focus_pair_appearance.py', f'reports/experiments/{RUN}-design.md')}


def inventory():
    if sha(OLD_DATA) != OLD_DATA_SHA:
        raise ValueError('Pinned complete candidate data required')
    evidence = json.loads(OLD_DATA.read_text())
    if any(sha(ROOT / p) != value for p, value in evidence['source_hashes'].items()):
        raise ValueError('Frozen original candidate data construction changed')
    balanced_path = REPORT / 'focus-balanced-candidate-ranker-v1-lomo.json'
    verification = REPORT / 'focus-balanced-candidate-ranker-v1-verification.json'
    if (sha(balanced_path) != 'dba19c4210c29941ca8056f49089ebac9d25eae8151c47931282af8366f5fe7c'
            or sha(verification) != '1f88a347bded62a83559c66881a5afabe3a20ccc912e830aa35b1e38769a1dad'):
        raise ValueError('Actual prior weighted experiment and verification required')
    balanced = json.loads(balanced_path.read_text())
    if any(sha(ROOT / p) != value for p, value in balanced['source_hashes'].items()):
        raise ValueError('Frozen prior weighted experiment changed')
    spec_path = ROOT / '.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    if sha(spec_path) != evidence['training_spec_sha256']:
        raise ValueError('Actual fitting feature specification changed')
    spec = json.loads(spec_path.read_text())
    if [r['stem'] for r in evidence['records']] != spec['contract']['fitting_stems']:
        raise ValueError('Exact original fitting-only movie inventory required')
    folders = [ROOT / '.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
               ROOT / '.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    records = []
    for old in evidence['records']:
        stem = old['stem']
        matches = [f for f in folders if (f / stem / 'manifest.json').exists()]
        if len(matches) != 1:
            raise ValueError('Unique original fitting feature root required')
        folder = matches[0]
        path = folder / stem / 'manifest.json'
        if sha(path) != old['manifest_sha256']:
            raise ValueError('Actual fitting feature manifest changed')
        manifest = json.loads(path.read_text())
        pairs = manifest['pairs']
        if (manifest['role'] != 'fitting' or manifest['stem'] != stem
                or [p['file'] for p in pairs] != [f'{t:03d}.npz' for t in range(99)]
                or [{'file': p['file'], 'sha256': p['sha256']} for p in pairs] != old['packet_hashes']):
            raise ValueError('All99 exact fitting-only feature packets required')
        records.append((old, folder, manifest))
    if len(records) != 12 or evidence['total_groups'] != 10915 or evidence['total_choices'] != 4691320:
        raise ValueError('Complete original twelve-movie data required')
    return evidence, spec['motion_parameters'], records


def main():
    started = time.monotonic()
    target = REPORT / f'{RUN}-data-smoke.json'
    if target.exists() or CACHE.exists():
        raise ValueError('Never overwrite partial or complete appearance data')
    frozen = sources()
    evidence, parameters, original = inventory()
    planned_bytes = evidence['total_choices'] * 64 * 4
    if planned_bytes > 2 * 1024 ** 3 or shutil.disk_usage(ROOT).free < 2 * planned_bytes:
        raise ValueError('Descriptor disk cap/headroom failed before allocation')
    stress = min((-p['source_nodes'] * (p['known_parent'] + p['known_absent']), r['stem'], p['source_frame'])
                 for r, _, m in original for p in m['pairs'] if p['known_parent'] and p['known_absent'])
    CACHE.mkdir()
    records = []
    for old, folder, manifest in original:
        tick = time.monotonic()
        stem = old['stem']
        base_path = ROOT / '.biohub/cache/focus-candidate-ranker-v1' / (stem + '.npz')
        if sha(base_path) != old['sha256']:
            raise ValueError('Actual original candidate rows changed')
        base = arrays(base_path)
        validate(base)
        raw_path = folder / 'raw_detections' / (stem + '.npz')
        if sha(raw_path) != old['raw_sha256']:
            raise ValueError('Actual fitting raw nodes changed')
        raw = arrays(raw_path)['coords']
        if sorted(set(raw[:, 0])) != list(range(100)):
            raise ValueError('Complete100frame fitting movie required')
        path = CACHE / (stem + '-appearance.npy')
        output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(old['choices'], 64))
        stats = empty_stats()
        cursor = groups = 0
        max_cosine_error = 0.
        for pair in manifest['pairs']:
            packet_path = folder / stem / pair['file']
            if sha(packet_path) != pair['sha256']:
                raise ValueError('Actual original feature packet changed')
            packet = arrays(packet_path)
            counts = validate_pair(packet, raw)
            if any(counts[k] != pair[k] for k in ('source_frame', 'source_nodes', 'target_nodes', 'known_parent', 'known_absent', 'unknown')):
                raise ValueError('Exact original geometry/label coverage differs')
            if not counts['known_parent'] + counts['known_absent']:
                continue
            data = pack(packet, parameters, 'fitting')
            expected = frame_packet(base, counts['source_frame'])
            if set(data) != set(expected) or any(not np.array_equal(data[k], expected[k]) for k in data):
                raise ValueError('All original features, labels and row identities must replay exactly')
            feature = descriptors(packet, np.flatnonzero(packet['labels'] >= 0))
            if feature.shape != (len(data['offset']), 64):
                raise ValueError('Complete appearance rows required')
            error = float(np.max(np.abs(feature[:, 32:].sum(axis=1) - data['features'][:, 7])))
            if error > 1e-6:
                raise ValueError('Original cosine must survive richer feature construction')
            max_cosine_error = max(error, max_cosine_error)
            output[cursor:cursor + len(feature)] = feature
            accumulate(stats, feature, data, 'fitting')
            cursor += len(feature)
            groups += len(data['starts'])
        if (cursor != old['choices'] or groups != old['groups']
                or stats['count'].tolist() != [cursor - groups - int(base['present'].sum()), int(base['present'].sum())]):
            raise ValueError('All descriptor rows and real-pair labels must be covered once')
        output.flush()
        del output
        restored = np.load(path, allow_pickle=False, mmap_mode='r')
        if restored.shape != (cursor, 64) or restored.dtype != np.float32 or np.any(restored[base['null_rows']] != 0):
            raise ValueError('Actual saved descriptor shape/null rows differ')
        moment_path = CACHE / (stem + '-moments.json')
        moment_path.write_text(json.dumps({k: v.tolist() for k, v in stats.items()}, indent=2, allow_nan=False))
        records.append(dict(stem=stem, descriptor_sha256=sha(path), moments_sha256=sha(moment_path),
                            choices=cursor, groups=groups, descriptor_bytes=cursor * 64 * 4,
                            real_pair_counts=stats['count'].tolist(), base_sha256=sha(base_path),
                            manifest_sha256=old['manifest_sha256'], raw_sha256=old['raw_sha256'],
                            maximum_cosine_error=max_cosine_error, elapsed_seconds=time.monotonic() - tick))
        del base, restored, raw
        print(json.dumps(records[-1]), flush=True)
    if (sum(r['choices'] for r in records) != 4691320 or sum(r['groups'] for r in records) != 10915
            or sum(r['real_pair_counts'][1] for r in records) != 10754):
        raise ValueError('Complete original data totals required')
    extraction_seconds = time.monotonic() - started
    stem, frame = stress[1:]
    base = arrays(ROOT / '.biohub/cache/focus-candidate-ranker-v1' / (stem + '.npz'))
    packet = frame_packet(base, frame)
    selected = np.flatnonzero(base['source_frame'] == frame)
    begin = int(base['starts'][selected[0]])
    end = int(base['starts'][selected[-1]] + base['sizes'][selected[-1]])
    stored = np.load(CACHE / (stem + '-appearance.npy'), mmap_mode='r', allow_pickle=False)
    feature = stored[begin:end].copy()
    stats = empty_stats()
    accumulate(stats, feature, packet, 'fitting')
    projection = projector(stats, 'fitting')
    samples = [(packet, feature)]
    smoke = []
    for arm, dim in [('full', 72), ('lda', 9)]:
        provider = lambda: blocks(samples, projection, arm)
        absent_weight = float(np.sqrt(packet['present'].sum() / (packet['present'] == 0).sum()))
        direction = np.arange(1., dim + 1)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        _, gradient = objective(np.zeros(dim), provider, absent_weight)
        numerical = (objective(eps * direction, provider, absent_weight)[0]
                     - objective(-eps * direction, provider, absent_weight)[0]) / (2 * eps)
        analytic = float(gradient @ direction)
        error = abs(numerical - analytic) / max(1., abs(analytic))
        if error > 2e-6:
            raise ValueError('Actual richer feature loss gradient failed')
        tick = time.monotonic()
        model = fit(samples, projection, arm, 'fitting')
        fit_seconds = time.monotonic() - tick
        model_path = CACHE / (arm + '-smoke-model.json')
        model_path.write_text(json.dumps(model, indent=2, allow_nan=False))
        restored = json.loads(model_path.read_text())
        if restored != model or metrics(samples, model) != metrics(samples, restored):
            raise ValueError('Actual richer model/prediction reload differs')
        smoke.append(dict(arm=arm, dimensions=dim, model_sha256=sha(model_path), fit_seconds=fit_seconds,
                          initial_objective=model['initial_objective'], objective=model['objective'],
                          iterations=model['iterations'], evaluations=model['evaluations'],
                          gradient_relative_error=error, exact_model_prediction_reload=True))
        print(json.dumps(dict(smoke=smoke[-1])), flush=True)
    if sources() != frozen:
        raise ValueError('Frozen appearance design or methods changed during run')
    result = dict(status='verified_complete_fitting_appearance_data_and_real_smokes', run_id=RUN,
                  source_hashes=frozen, old_data_sha256=OLD_DATA_SHA, records=records,
                  total_groups=10915, total_choices=4691320, descriptor_bytes=planned_bytes,
                  real_pair_counts=[4669651, 10754], extraction_seconds=extraction_seconds,
                  stress=dict(stem=stem, source_frame=frame, real_choices=-stress[0], groups=len(packet['starts']),
                              projection=projection, arms=smoke), quality_evaluated=False,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  gpu_seconds=0, authorized_for_submission=False, elapsed_seconds=time.monotonic() - started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'stress', 'source_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       env=dict(os.environ), timeout=900, check=True)
    else:
        raise ValueError('Unsupported arguments')
