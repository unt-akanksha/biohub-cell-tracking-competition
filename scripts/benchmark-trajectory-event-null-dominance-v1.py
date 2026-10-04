"""Compare exact event optima on frozen cases; no training, GT, or deployment."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha
from research.trajectory_event_assignment_v1 import problem, solve
from research.trajectory_event_vectorized_dominance_v1 import allowed_options as previous
from research.trajectory_event_null_dominance_v1 import allowed_options


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(smoke):
    started = time.monotonic()
    name = 'trajectory-event-null-dominance-v1-' + ('smoke' if smoke else 'full')
    receipt = ROOT / 'reports/experiments' / (name + '.json')
    assert not receipt.exists()
    evaluator = runpy.run_path(str(ROOT / 'scripts/evaluate-trajectory-event-fork16-held-v1.py'))
    contracts = evaluator['fixed_fit_contracts'](ROOT)
    fit_path = ROOT / 'reports/experiments/trajectory-event-fork16-fit-44b6-v1.json'
    fit = read(fit_path)
    assert fit['status'] == 'source_event_training_complete'
    weights_path = ROOT / '.biohub/cache/trajectory-event-fork16-fit-44b6-v1/final-weights.npz'
    assert sha(weights_path) == fit['weights_sha256']
    with np.load(weights_path, allow_pickle=False) as model:
        weights = model['weights'].copy()
    # Fixed cases from opposite source: first, midpoint, largest vocabulary.
    contract = read(ROOT / '.biohub/cache/trajectory-event-fork16-fit-6bba-v1/CONTRACT.json')
    records = contract['case_records']
    indices = [0] if smoke else sorted({0, len(records) // 2, max(range(len(records)), key=lambda i: records[i]['options'])})
    if not smoke:
        proof = read(ROOT / 'reports/experiments/trajectory-event-null-dominance-v1-smoke.json')
        assert proof['status'] == 'exact_event_optima_verified'
        assert proof['source_sha256'] == sha(Path(__file__))
        assert proof['helper_sha256'] == sha(ROOT / 'research/trajectory_event_null_dominance_v1.py')
    result = dict(status='benchmark_running', source_sha256=sha(Path(__file__)),
        helper_sha256=sha(ROOT / 'research/trajectory_event_null_dominance_v1.py'),
        fit_contracts=contracts, model_sha256=sha(weights_path), cases=indices, records=[],
        source_labels_loaded=False, optimizer_or_live_jobs_changed=False, gpu_used=False,
        production_changed=False, complete_movie_graph_parity_established=False,
        no_end_to_end_speedup_claim=True, authorized_for_submission=False)

    def persist():
        result['seconds'] = time.monotonic() - started
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    persist()
    try:
        for index in indices:
            row = records[index]
            path = ROOT / '.biohub/cache' / row['path']
            assert sha(path) == row['sha256']
            with np.load(path, allow_pickle=False) as data:
                # No target, margin, or allowed-supervision arrays are read.
                case = problem(data['options'], int(data['nparents']), int(data['nchildren']))
                features = data['features'].astype(np.float64)
            for model_name, values in (('anchor', np.asarray(contract['anchor'])), ('final44', weights)):
                scores = features @ values
                old_start = time.perf_counter()
                old_mask, old_report = previous(case, scores)
                old_solution = solve(case, scores, allowed=old_mask, time_limit=10.)
                old_seconds = time.perf_counter() - old_start
                new_start = time.perf_counter()
                new_mask, new_report = allowed_options(case, scores)
                new_solution = solve(case, scores, allowed=new_mask, time_limit=10.)
                new_seconds = time.perf_counter() - new_start
                delta = float(scores @ (new_solution.astype(float) - old_solution))
                exact = bool(np.array_equal(old_solution, new_solution))
                result['records'].append(dict(case_index=index, stem=row['stem'], t=row['t'],
                    case_sha256=row['sha256'], model=model_name, original=old_report, pruned=new_report,
                    objective_delta=delta, exact_assignment=exact,
                    original_seconds=old_seconds, pruned_seconds=new_seconds))
                persist()
                assert abs(delta) <= 1e-8, 'Pruning changed the optimum'
                print(json.dumps(result['records'][-1]), flush=True)
        assert evaluator['fixed_fit_contracts'](ROOT) == contracts
        old_total = sum(r['original_seconds'] for r in result['records'])
        new_total = sum(r['pruned_seconds'] for r in result['records'])
        result.update(status='exact_event_optima_verified', all_assignments_exact=all(r['exact_assignment'] for r in result['records']),
            original_seconds=old_total, pruned_seconds=new_total, stage_speed_ratio=old_total / new_total)
        persist()
    except BaseException as error:
        result.update(status='failed_requires_inspection', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    with threadpool_limits(limits=1, user_api='blas'):
        main(parser.parse_args().smoke)
