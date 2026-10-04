"""Small real fitting-only history-head optimizer test; no quality experiment."""
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
from research.focus_pair_history_head import (pack, validate, objective, class_weights,
    hessian, blocks, fit, metrics, CONSTRAINED, UPPER)
from research.focus_candidate_ranker import pack as base_pack, validate as base_validate

RUN = 'focus-pair-history-head-v1'
CACHE = ROOT/'.biohub/cache'/RUN
SOURCES = ['scripts/smoke-focus-pair-history-head.py', 'research/focus_pair_history_head.py',
    'research/focus_pair_history.py', 'research/focus_candidate_ranker.py',
    'research/focus_balanced_candidate_ranker.py', 'research/focus_pair_appearance_conditioning.py',
    'tests/test_focus_pair_history_head.py', f'reports/experiments/{RUN}-design.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    target = ROOT/f'reports/experiments/{RUN}-smoke.json'
    if CACHE.exists() or target.exists():
        raise ValueError('Never overwrite partial or completed smoke')
    frozen = {p: sha(ROOT/p) for p in SOURCES}
    audit_path = ROOT/'reports/experiments/focus-pair-history-v1-audit.json'
    if sha(audit_path) != '799102689148f65daeb557a1aaf8fc002f2524fa5ccb2c2efaf527c4ee81432d':
        raise ValueError('Completed original history data/functionality audit required')
    audit = json.loads(audit_path.read_text())
    if any(sha(ROOT/p) != v for p, v in audit['source_hashes'].items()):
        raise ValueError('Original history method must remain unchanged')
    root = ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs/6bba_57b7cc1e'
    current_path, previous_path = root/'031.npz', root/'030.npz'
    if (sha(current_path) != audit['smoke']['current_packet_sha256']
            or sha(previous_path) != audit['smoke']['previous_packet_sha256']):
        raise ValueError('Exact original stress/history packets required')
    with np.load(current_path, allow_pickle=False) as saved:
        current = {k: saved[k].copy() for k in saved.files}
    with np.load(previous_path, allow_pickle=False) as saved:
        previous = {k: saved[k].copy() for k in saved.files if k != 'labels'}
    spec_path = ROOT/'.biohub/cache/kernel-outputs/focus-parent-dropout-training-v1/focus_parent_dropout_training/runtime/training_spec.json'
    if sha(spec_path) != audit['training_spec_sha256']:
        raise ValueError('Existing physical parameters and fitting scope required')
    spec = json.loads(spec_path.read_text())
    if '6bba_57b7cc1e' not in spec['contract']['fitting_stems']:
        raise ValueError('Functionality packet must be an original fitting movie')
    parameters = spec['motion_parameters']
    before = [{k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in p.items()} for p in (current, previous)]
    data = pack(current, previous, parameters, 'fitting')
    ids = validate(data)
    weights, aw = class_weights(data, 'fitting')
    if (len(data['starts']), int(data['present'].sum()), len(data['offset'])) != (20, 19, 23220):
        raise ValueError('Exact original20 known groups and all23,220 choices required')
    old = base_pack(current, parameters, 'fitting')
    if any(not np.array_equal(data[k][:, :8] if k == 'features' else data[k], old[k]) for k in old):
        raise ValueError('Original features/offsets/groups/labels changed')
    theta8 = np.arange(8, dtype=float)*.01
    old_value, old_grad = objective(theta8, old, base_validate(old), weights)
    expanded_value, expanded_grad = objective(np.r_[theta8, np.zeros(7)], data, ids, weights)
    if abs(old_value-expanded_value) > 1e-9 or not np.allclose(old_grad, expanded_grad[:8], atol=1e-8, rtol=1e-12):
        raise ValueError('Zero-history-weight original objective is not equivalent')
    theta, direction = np.zeros(15), np.arange(1., 16.)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    initial, gradient = objective(theta, data, ids, weights)
    plus = objective(theta+eps*direction, data, ids, weights)
    minus = objective(theta-eps*direction, data, ids, weights)
    numeric = (plus[0]-minus[0])/(2*eps)
    gradient_error = abs(numeric-float(gradient @ direction))/max(1., abs(float(gradient @ direction)))
    curvature, _ = hessian(theta, lambda: blocks(data), aw, 'fitting')
    numeric_hv = (plus[1]-minus[1])/(2*eps)
    expected_hv = curvature @ direction
    curvature_error = float(np.linalg.norm(numeric_hv-expected_hv)/max(1., np.linalg.norm(expected_hv)))
    if max(gradient_error, curvature_error) > 2e-6:
        raise ValueError('Real analytic gradient/Hessian check failed')
    CACHE.mkdir()
    tick = time.monotonic()
    model, execution = fit(data, CACHE/'fit', 'fitting')
    fit_seconds = time.monotonic()-tick
    saved = json.loads((CACHE/'fit/model.json').read_text())
    if (saved != model or metrics(data, saved) != metrics(data, model)
            or not model['objective'] < initial or np.any(np.asarray(model['theta'])[CONSTRAINED] > UPPER)):
        raise ValueError('Actual fitted loss/bounds/model/predictions failed')
    for snapshot, packet in zip(before, (current, previous)):
        if any(hashlib.sha256(packet[k].tobytes()).hexdigest() != v for k, v in snapshot.items()):
            raise ValueError('Actual optimizer packing mutated inputs')
    if any(sha(ROOT/p) != v for p, v in frozen.items()):
        raise ValueError('Frozen small-test implementation changed')
    result = dict(status='verified_real_history_head_optimizer_smoke', run_id=RUN,
        source_hashes=frozen, history_audit_sha256=sha(audit_path),
        current_packet_sha256=sha(current_path), previous_packet_sha256=sha(previous_path),
        groups=20, parents=19, absent=1, choices=23220, features=15, gradient_relative_error=gradient_error,
        hessian_relative_error=curvature_error, original_objective_equivalent=True,
        model_sha256=sha(CACHE/'fit/model.json'), initial_objective=initial,
        final_objective=model['objective'], iterations=model['iterations'], evaluations=model['evaluations'],
        training_metrics=metrics(data, model), execution=execution, fit_seconds=fit_seconds,
        exact_model_prediction_reload=True, inputs_unchanged=True, quality_evaluated=False,
        gpu_seconds=0, diagnostic_movies_opened=0, source_movies_opened=0, new_target_movies_opened=0,
        authorized_for_submission=False, elapsed_seconds=time.monotonic()-started, actual_worker_pid=os.getpid())
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        main()
    elif not sys.argv[1:]:
        packages = str(Path(sys.prefix)/'Lib/site-packages')
        source = str(Path(__file__).resolve())
        code = (f'import sys,runpy; sys.path.insert(0,{packages!r}); '
                f'sys.argv=[{source!r},"--worker"]; runpy.run_path(sys.argv[0],run_name="__main__")')
        subprocess.run([sys._base_executable, '-S', '-c', code], cwd=ROOT, timeout=180, check=True)
    else:
        raise ValueError('Unsupported arguments')
