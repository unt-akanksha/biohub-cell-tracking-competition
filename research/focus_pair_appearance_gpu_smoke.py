"""One fixed real fitting packet, two CPU/GPU equivalence smoke fits."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from research.focus_candidate_ranker import pack
from research.focus_pair_appearance import descriptors
from research.focus_pair_appearance_head import blocks, objective, metrics
from research.focus_pair_appearance_cuda import CudaObjective


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(expected, actual):
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-7, atol=2e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-7, atol=2e-6)
    return dict(loss_absolute_error=abs(actual[0] - expected[0]),
                gradient_max_absolute_error=float(np.max(np.abs(actual[1] - expected[1]))))


def choices(scores, starts, sizes):
    return np.array([np.argmax(scores[s:s + n]) for s, n in zip(starts, sizes)], np.int64)


def synthetic_check():
    rng = np.random.default_rng(274919)
    sizes = np.array([1, 4, 3], np.int64)
    starts = np.array([0, 1, 5], np.int64)
    chosen = np.array([0, 2, 7], np.int64)
    x = rng.normal(size=(8, 9))
    x[[0, 4, 7]] = 0
    offset = rng.normal(size=8)
    offset[[0, 4, 7]] = -4.5
    parent = np.array([0, 1, 0], np.int64)
    block = dict(x=x, offset=offset, starts=starts, sizes=sizes, chosen=chosen,
                 present=parent, ids=np.repeat(np.arange(3), sizes))
    cuda = CudaObjective(x, offset, sizes, chosen, parent, 2., 'fitting')
    theta = rng.normal(size=9) * .1
    result = compare(objective(theta, lambda: iter([block]), 2.), cuda(theta))
    return dict(groups=3, choices=8, includes_null_only_group=True, comparison=result)


def main():
    started = time.monotonic()
    runtime = Path(__file__).resolve().parent.parent
    output = runtime.parent / 'outputs'
    output.mkdir(exist_ok=False)
    spec = json.loads((runtime / 'spec.json').read_text())
    expected_sources = json.loads((runtime / 'source_hashes.json').read_text())
    if any(sha(runtime / name) != value for name, value in expected_sources.items()):
        raise ValueError('Exact embedded runtime, inputs and controls required')
    with np.load(runtime / 'pair.npz', allow_pickle=False) as saved:
        packet = {k: saved[k].copy() for k in saved.files}
    if int(packet['source_frame']) != 31:
        raise ValueError('Fixed audited fitting frame required')
    base = pack(packet, spec['motion_parameters'], 'fitting')
    appearance = descriptors(packet, np.flatnonzero(packet['labels'] >= 0))
    if len(base['starts']) != 20 or len(base['offset']) != 23220 or int(base['present'].sum()) != 19:
        raise ValueError('Exact complete real smoke groups required')
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    synthetic = synthetic_check()
    records = []
    for arm in ('full', 'lda'):
        cpu = json.loads((runtime / (arm + '-cpu-model.json')).read_text())
        samples = [(base, appearance)]
        control = metrics(samples, cpu)
        for key in ('known_parent', 'known_absent', 'correct_parent', 'correct_absent'):
            if control[key] != spec['controls'][arm][key]:
                raise ValueError('Host CPU reference decisions must replay before CUDA optimization')
        if abs(control['loss_sum'] - spec['controls'][arm]['loss_sum']) > 2e-6:
            raise ValueError('Host CPU reference posterior must replay')
        provider = lambda: blocks(samples, cpu['projection'], arm)
        prepared = list(provider())
        if len(prepared) != 1:
            raise ValueError('One complete preselected frame block required')
        block = prepared[0]
        cuda = CudaObjective(block['x'], block['offset'], block['sizes'], block['chosen'], block['present'],
                             cpu['absent_weight'], 'fitting')
        theta = np.asarray(cpu['theta'])
        comparisons = [compare(objective(t, provider, cpu['absent_weight']), cuda(t))
                       for t in (np.zeros(len(theta)), theta)]
        for t, comparison in zip((np.zeros(len(theta)), theta), comparisons):
            cpu_scores = block['offset'] + block['x'] @ t
            gpu_scores = (cuda.offset + cuda.x @ torch.as_tensor(t, dtype=torch.float64, device='cuda:0')).cpu().numpy()
            np.testing.assert_allclose(gpu_scores, cpu_scores, rtol=2e-7, atol=2e-6)
            if not np.array_equal(choices(cpu_scores, block['starts'], block['sizes']),
                                  choices(gpu_scores, block['starts'], block['sizes'])):
                raise ValueError('Every complete CPU/CUDA argmax must agree')
            comparison['maximum_logit_error'] = float(np.max(np.abs(cpu_scores - gpu_scores)))
            comparison['all_target_argmax_equal'] = True
        bounds = [(None, None)] * len(theta)
        bounds[4:7] = [(None, .5)] * 3
        tick = time.monotonic()
        fitted = minimize(cuda, np.zeros(len(theta)), jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
        fit_seconds = time.monotonic() - tick
        if not fitted.success or not np.isfinite(fitted.x).all() or abs(fitted.fun - cpu['objective']) > 1e-5:
            raise ValueError(f'GPU smoke optimizer/CPU objective equivalence failed: {fitted.message}')
        model = dict(cpu, theta=fitted.x.tolist(), objective=float(fitted.fun),
                     iterations=int(fitted.nit), evaluations=int(fitted.nfev))
        path = output / (arm + '-smoke-model.json')
        path.write_text(json.dumps(model, indent=2, allow_nan=False))
        restored = json.loads(path.read_text())
        actual = metrics(samples, restored)
        if restored != model or actual != metrics(samples, model):
            raise ValueError('Exact GPU fitted model/prediction reload differs')
        control_choices = choices(block['offset'] + block['x'] @ theta, block['starts'], block['sizes'])
        fitted_choices = choices(block['offset'] + block['x'] @ fitted.x, block['starts'], block['sizes'])
        if not np.array_equal(control_choices, fitted_choices):
            raise ValueError('Every optimized smoke decision must agree with the CPU fit')
        for key in ('known_parent', 'known_absent', 'correct_parent', 'correct_absent'):
            if actual[key] != control[key]:
                raise ValueError('GPU smoke must reproduce discrete CPU decisions')
        records.append(dict(arm=arm, objective_comparisons=comparisons, cpu_control=control,
                            fitted_metrics=actual, model_sha256=sha(path), gpu_fit_seconds=fit_seconds,
                            objective=float(fitted.fun), cpu_objective=cpu['objective'],
                            iterations=int(fitted.nit), evaluations=int(fitted.nfev),
                            all_optimized_target_choices_equal_cpu=True,
                            target_choices_sha256=hashlib.sha256(fitted_choices.tobytes()).hexdigest(),
                            exact_parameter_prediction_reload=True))
        print(json.dumps(records[-1]), flush=True)
        del cuda, prepared, block
        torch.cuda.empty_cache()
    if any(sha(runtime / name) != value for name, value in expected_sources.items()):
        raise ValueError('Runtime changed during GPU smoke')
    result = dict(status='completed_pair_appearance_gpu_compatibility_smoke', run_id=spec['run_id'],
                  source_hashes=expected_sources, spec_sha256=sha(runtime / 'spec.json'), records=records,
                  synthetic=synthetic, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  gpu_count=torch.cuda.device_count(), peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                  elapsed_seconds=time.monotonic() - started, diagnostic_movies_opened=0,
                  source_movies_opened=0, new_target_movies_opened=0,
                  quality_evaluated=False, authorized_for_submission=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_hashes', 'records')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
