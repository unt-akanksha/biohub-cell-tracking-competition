"""Replay downloaded GPU smoke artifacts on CPU; never refit or launch."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.focus_candidate_ranker import pack
from research.focus_pair_appearance import descriptors
from research.focus_pair_appearance_head import blocks, objective, metrics
from research.focus_pair_appearance_gpu_smoke import choices

RUN = 'focus-pair-appearance-gpu-smoke-v1'
CACHE = ROOT / '.biohub/cache/kernel-outputs' / RUN / 'focus_pair_appearance_gpu_smoke'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / 'reports/experiments' / f'{RUN}-verification.json'
    if target.exists():
        raise ValueError('Never overwrite a completed GPU verification')
    stage = ROOT / 'kaggle' / ('biohub-' + RUN)
    identity = json.loads((stage / 'staged_identity.json').read_text())
    if (sha(stage / ('biohub-' + RUN + '.ipynb')) != identity['notebook_sha256']
            or sha(stage / 'kernel-metadata.json') != identity['metadata_sha256']
            or sha(ROOT / 'scripts/build-focus-pair-appearance-gpu-smoke.py') != identity['builder_sha256']):
        raise ValueError('Original launched stage identity changed')
    builder = runpy.run_path(str(ROOT / 'scripts/build-focus-pair-appearance-gpu-smoke.py'))
    notebook, metadata = builder['build']()
    if (notebook != json.loads((stage / metadata['code_file']).read_text(encoding='utf-8'))
            or metadata != json.loads((stage / 'kernel-metadata.json').read_text(encoding='utf-8'))):
        raise ValueError('Exact frozen notebook and source must rebuild')
    runtime = CACHE / 'runtime'
    hashes = json.loads((runtime / 'source_hashes.json').read_text())
    if any(sha(runtime / name) != value for name, value in hashes.items()):
        raise ValueError('Actual returned runtime/input/control bytes differ')
    spec = json.loads((runtime / 'spec.json').read_text())
    result = json.loads((CACHE / 'outputs/result.json').read_text())
    terminal = json.loads((CACHE / 'launcher_terminal.json').read_text())
    if (result['status'] != 'completed_pair_appearance_gpu_compatibility_smoke'
            or result['run_id'] != RUN or result['source_hashes'] != hashes
            or result['spec_sha256'] != sha(runtime / 'spec.json')
            or result['quality_evaluated'] is not False or result['authorized_for_submission'] is not False
            or any(result[k] != 0 for k in ('diagnostic_movies_opened', 'source_movies_opened', 'new_target_movies_opened'))
            or terminal['status'] != 'completed' or terminal['declared_budget_seconds'] != 300
            or terminal['submission_performed'] is not False or not 0 < terminal['elapsed_seconds'] < 240
            or not 0 < result['elapsed_seconds'] <= terminal['elapsed_seconds']
            or not result['synthetic']['includes_null_only_group'] or result['gpu'] != 'Tesla T4'):
        raise ValueError('Complete actual fitting-only bounded GPU result required')
    with np.load(runtime / 'pair.npz', allow_pickle=False) as saved:
        packet = {k: saved[k].copy() for k in saved.files}
    base = pack(packet, spec['motion_parameters'], 'fitting')
    appearance = descriptors(packet, np.flatnonzero(packet['labels'] >= 0))
    samples = [(base, appearance)]
    records = []
    if [r['arm'] for r in result['records']] != ['full', 'lda']:
        raise ValueError('Both original arms must finish')
    for row in result['records']:
        arm = row['arm']
        cpu = json.loads((runtime / (arm + '-cpu-model.json')).read_text())
        model_path = CACHE / 'outputs' / (arm + '-smoke-model.json')
        model = json.loads(model_path.read_text())
        if sha(model_path) != row['model_sha256']:
            raise ValueError('Actual GPU model identity differs')
        mutable = {'theta', 'objective', 'iterations', 'evaluations'}
        if ({k: v for k, v in model.items() if k not in mutable}
                != {k: v for k, v in cpu.items() if k not in mutable}):
            raise ValueError('GPU may only change fitted numerical coefficients/solver metadata')
        provider = lambda: blocks(samples, model['projection'], arm)
        value, gradient = objective(np.asarray(model['theta']), provider, model['absent_weight'])
        actual = metrics(samples, model)
        control = metrics(samples, cpu)
        for key in ('known_parent', 'known_absent', 'correct_parent', 'correct_absent'):
            if actual[key] != row['fitted_metrics'][key] or actual[key] != control[key]:
                raise ValueError('Every CPU/GPU parent/absence count must replay')
        if (abs(value - model['objective']) > 1e-7 or abs(value - cpu['objective']) > 1e-5
                or abs(actual['loss_sum'] - row['fitted_metrics']['loss_sum']) > 1e-7
                or model['objective'] != row['objective'] or model['iterations'] != row['iterations']
                or model['evaluations'] != row['evaluations']):
            raise ValueError('Actual saved CUDA objective and posterior must replay on CPU')
        block = next(provider())
        selected = choices(block['offset'] + block['x'] @ np.asarray(model['theta']), block['starts'], block['sizes'])
        original = choices(block['offset'] + block['x'] @ np.asarray(cpu['theta']), block['starts'], block['sizes'])
        if not np.array_equal(selected, original) or hashlib.sha256(selected.tobytes()).hexdigest() != row['target_choices_sha256']:
            raise ValueError('Every individual target choice must replay exactly')
        records.append(dict(arm=arm, objective=value, gradient_inf_norm=float(np.abs(gradient).max()),
                            metrics=actual, all_target_choices_replayed=True, model_sha256=sha(model_path)))
    verified = dict(status='host_verified_pair_appearance_gpu_smoke', kernel='indarkarhana/biohub-' + RUN,
                    kernel_version=1, staged_identity=identity, result_sha256=sha(CACHE / 'outputs/result.json'),
                    verifier_sha256=sha(Path(__file__)), source_hashes=hashes, records=records,
                    launcher=terminal, worker_elapsed_seconds=result['elapsed_seconds'],
                    gpu_fit_seconds=[r['gpu_fit_seconds'] for r in result['records']],
                    peak_allocated_bytes=result['peak_allocated_bytes'], quality_evaluated=False,
                    authorized_for_submission=False, host_optimizer_refit=False)
    target.write_text(json.dumps(verified, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in verified.items() if k not in ('source_hashes', 'staged_identity')}, indent=2))


if __name__ == '__main__':
    main()
