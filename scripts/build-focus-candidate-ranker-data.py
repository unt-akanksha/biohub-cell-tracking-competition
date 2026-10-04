"""Audit full fitting candidates, persist compact choices, run one real smoke."""
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
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_cached_pair import validate_pair
from research.focus_candidate_ranker import pack, combine, validate, objective, fit, metrics, MAX_BYTES
from research.focus_parent_presence import metrics as neural_metrics

RUN = 'focus-candidate-ranker-v1'
FILES = ['scripts/build-focus-candidate-ranker-data.py', 'research/focus_candidate_ranker.py',
         'research/focus_cached_pair.py', f'reports/experiments/{RUN}-design.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    cache = ROOT / '.biohub/cache' / RUN
    target = ROOT / f'reports/experiments/{RUN}-data-smoke.json'
    if cache.exists() or target.exists():
        raise ValueError('Never overwrite complete or partial candidate data')
    frozen = {p: sha(ROOT / p) for p in FILES}
    receipt_path = ROOT / 'reports/experiments/focus-parent-dropout-training-v1-result.json'
    if sha(receipt_path) != '854d43703018ff135cfe464c5597903df484fa9ac52d4fc3fc6eaa71f4c1c623':
        raise ValueError('Exact original verified feature-training provenance required')
    old = json.loads(receipt_path.read_text())
    spec_path = ROOT / '.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    if sha(spec_path) != old['worker']['training_spec_sha256']:
        raise ValueError('Actual upstream feature specification changed')
    spec = json.loads(spec_path.read_text())
    neural_data, evidence = runpy.run_path(str(ROOT / 'scripts/fit-focus-quadratic-presence.py'))['load_fitting']()
    stems = spec['contract']['fitting_stems']
    if list(neural_data) != stems or len(stems) != 12:
        raise ValueError('Exact twelve fitting movies required')
    parameters = spec['motion_parameters']
    roots = [ROOT / '.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',
             ROOT / '.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    inventory = []
    for group, folder in zip(spec['feature_groups'], roots):
        for record in group['feature_records']:
            if record['role'] != 'fitting':
                continue
            stem = record['stem']
            manifest_path = folder / stem / 'manifest.json'
            if sha(manifest_path) != record['manifest_sha256']:
                raise ValueError('Actual fitting manifest changed')
            manifest = json.loads(manifest_path.read_text())
            if [p['file'] for p in manifest['pairs']] != [f'{t:03d}.npz' for t in range(99)]:
                raise ValueError('All99 fitting transitions required')
            inventory.append((stem, folder, record, manifest))
    if [r[0] for r in inventory] != stems:
        raise ValueError('Exact fitting-role inventory required')
    planned_groups = sum(p['known_parent'] + p['known_absent'] for _, _, _, m in inventory for p in m['pairs'])
    planned_choices = sum((p['known_parent'] + p['known_absent']) * (p['source_nodes'] + 1) for _, _, _, m in inventory for p in m['pairs'])
    estimate = planned_choices * 9 * 8 + planned_groups * 7 * 8
    if planned_groups != 10915 or estimate > MAX_BYTES:
        raise ValueError('Exact full candidate coverage must fit declared2GiB array budget')
    stress = max(((p['source_nodes'] * (p['known_parent'] + p['known_absent']), stem, int(p['source_frame']))
                  for stem, _, _, m in inventory for p in m['pairs']), key=lambda row: row[0])
    cache.mkdir()
    records, stress_data, null_losses = [], None, []
    for stem, folder, record, manifest in inventory:
        raw = folder / 'raw_detections' / (stem + '.npz')
        if sha(raw) != record['raw_sha256']:
            raise ValueError('Actual raw fitting coordinates changed')
        with np.load(raw, allow_pickle=False) as saved:
            coords = saved['coords'].copy()
        if sorted(set(coords[:, 0])) != list(range(100)):
            raise ValueError('Complete100frame fitting movie required')
        parts, hashes = [], []
        for t, pair in enumerate(manifest['pairs']):
            path = folder / stem / pair['file']
            if sha(path) != pair['sha256']:
                raise ValueError('Actual full-candidate packet changed')
            with np.load(path, allow_pickle=False) as saved:
                packet = {k: saved[k].copy() for k in saved.files}
            counts = validate_pair(packet, coords)
            if any(counts[k] != pair[k] for k in ('source_frame', 'source_nodes', 'target_nodes', 'known_parent', 'known_absent', 'unknown')):
                raise ValueError('Full packet geometry/count replay differs')
            hashes.append(dict(file=pair['file'], sha256=pair['sha256']))
            if not counts['known_parent'] + counts['known_absent']:
                continue
            data = pack(packet, parameters, 'fitting')
            validate(data)
            parts.append(data)
            if (stem, t) == (stress[1], stress[2]):
                stress_data = data
        combined = combine(parts)
        reference = neural_data[stem]
        if any(not np.array_equal(combined[k], reference[k]) for k in ('present', 'target_indices', 'source_frame')):
            raise ValueError('Candidate groups differ from verified original known-label identities')
        ids = validate(combined)
        maximum = np.maximum.reduceat(combined['offset'], combined['starts'])
        lse = maximum + np.log(np.add.reduceat(np.exp(combined['offset'] - maximum[ids]), combined['starts']))
        null_losses.extend((lse - combined['offset'][combined['chosen']])[combined['present'] == 0].tolist())
        path = cache / (stem + '.npz')
        np.savez_compressed(path, **combined)
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(combined) or any(not np.array_equal(saved[k], combined[k]) for k in combined):
                raise ValueError('Persisted full candidate choices differ')
        records.append(dict(stem=stem, sha256=sha(path), groups=len(combined['starts']), choices=len(combined['offset']),
                            bytes=sum(a.nbytes for a in combined.values()), physical=metrics(combined), neural=neural_metrics(reference),
                            raw_sha256=sha(raw), manifest_sha256=record['manifest_sha256'], packet_hashes=hashes))
        print(json.dumps({k: records[-1][k] for k in ('stem', 'groups', 'choices', 'bytes')}), flush=True)
    if (sum(r['choices'] for r in records) != planned_choices or sum(r['groups'] for r in records) != planned_groups
            or len(null_losses) != 161 or abs(np.mean(null_losses) - .961620872216142) > 2e-6
            or sum(r['physical']['correct_absent'] for r in records) != 115):
        raise ValueError('Complete coverage or original physical-null replay differs')
    if stress_data is None:
        raise ValueError('Actual preselected stress packet missing')
    ids = validate(stress_data)
    theta = np.zeros(8)
    value, gradient = objective(theta, stress_data, ids)
    direction = np.arange(1., 9.)
    direction /= np.linalg.norm(direction)
    step = 1e-6
    numerical = (objective(theta + step * direction, stress_data, ids)[0]
                 - objective(theta - step * direction, stress_data, ids)[0]) / (2 * step)
    analytic = float(gradient @ direction)
    error = abs(numerical - analytic)
    if not np.isfinite(value) or not np.isfinite(gradient).all() or error > 2e-5 * max(1., abs(analytic)):
        raise ValueError('Actual stress-packet finite-difference gradient failed')
    fit_started = time.monotonic()
    model = fit(stress_data, 'fitting')
    model_path = cache / 'stress-model.json'
    model_path.write_text(json.dumps(model, indent=2, allow_nan=False))
    restored = json.loads(model_path.read_text())
    if metrics(stress_data, model) != metrics(stress_data, restored):
        raise ValueError('Actual stress model posterior/decision roundtrip failed')
    smoke = dict(stem=stress[1], source_frame=stress[2], real_candidate_rows=stress[0],
                 model_sha256=sha(model_path), initial_objective=value, final_objective=model['objective'],
                 finite_difference_error=error, analytic_directional_derivative=analytic,
                 iterations=model['iterations'], evaluations=model['evaluations'], elapsed_seconds=time.monotonic() - fit_started,
                 exact_parameter_roundtrip=True, quality_evaluated=False)
    if any(sha(ROOT / p) != digest for p, digest in frozen.items()):
        raise ValueError('Frozen candidate-ranking method changed during extraction')
    result = dict(run_id=RUN, status='verified_fitting_candidate_data_and_real_smoke', source_hashes=frozen,
                  upstream_training_receipt_sha256=sha(receipt_path), training_spec_sha256=sha(spec_path),
                  original_summary_evidence=evidence, motion_parameters=parameters, records=records,
                  total_groups=planned_groups, total_choices=planned_choices, declared_array_bytes=estimate,
                  physical_null_nll=float(np.mean(null_losses)), stress_smoke=smoke, gpu_seconds=0,
                  diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
                  authorized_for_submission=False, elapsed_seconds=time.monotonic() - started)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], smoke=smoke, groups=planned_groups, choices=planned_choices,
                         elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                       env=dict(os.environ), timeout=300, check=True)
    else:
        raise ValueError('Unsupported arguments')
