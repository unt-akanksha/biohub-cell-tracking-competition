"""Complete-movie exactness test for experimental birth/death dominance."""
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
from research.trajectory_runtime_v1 import sha, validate_graph
from research.trajectory_event_release_v1 import check_event_stage
from research.trajectory_event_vectorized_inference_v1 import private_function
from research.trajectory_event_null_dominance_v1 import allowed_options


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(smoke):
    started = time.monotonic()
    name = 'trajectory-event-null-portable-v1-' + ('smoke' if smoke else 'full')
    reports = ROOT / 'reports/experiments'
    receipt = reports / (name + '.json')
    target = ROOT / '.biohub/cache' / name
    assert not target.exists() and not receipt.exists()
    reference_name = 'trajectory-event-fork16-learned-44b6-portable-v1'
    proof_path = reports / (reference_name + '.json')
    proof = read(proof_path)
    assert sha(proof_path) == '0f35ed01f9aa14b1fd3856aacc2e3e61a86d74c7b9f95bf2185a4b41073525a2'
    assert proof['status'] == 'learned_portable_smoke_exact' and not proof['ground_truth_opened']
    benchmark_path = reports / 'trajectory-event-null-dominance-v1-full.json'
    benchmark = read(benchmark_path)
    assert benchmark['status'] == 'exact_event_optima_verified' and benchmark['all_assignments_exact']
    assert benchmark['helper_sha256'] == sha(ROOT / 'research/trajectory_event_null_dominance_v1.py')
    if not smoke:
        small = read(reports / 'trajectory-event-null-portable-v1-smoke.json')
        assert small['status'] == 'complete_portable_graphs_exact' and small['source_sha256'] == sha(Path(__file__))
    code_path = ROOT / '.biohub/cache/trajectory-event-fork16-vectorized-portable-v1/event-trajectory.py'
    assert sha(code_path) == proof['code_sha256']
    runtime = runpy.run_path(str(code_path))
    original_pruner = runtime['infer'].__globals__['allowed_options']
    infer = private_function(runtime['infer'], {'allowed_options': allowed_options})
    prepared = private_function(runtime['refine_prepared'], {'infer': infer})
    refine = private_function(runtime['refine'], {'refine_prepared': prepared})
    assert runtime['infer'].__globals__['allowed_options'] is original_pruner
    model_path = ROOT / '.biohub/cache' / reference_name / 'event-trajectory-model.json'
    assert sha(model_path) == proof['model_sha256']
    model = read(model_path)
    original = read(ROOT / proof['original_smoke_path'])
    movies = proof['movies'][:1] if smoke else proof['movies']
    target.mkdir()
    result = dict(status='replaying_complete_movies', source_sha256=sha(Path(__file__)),
        helper_sha256=benchmark['helper_sha256'], benchmark_sha256=sha(benchmark_path),
        reference_sha256=sha(proof_path), model_sha256=sha(model_path), records=[],
        ground_truth_opened=False, model_changed=False, live_jobs_changed=False,
        staged_worker_changed=False, gpu_used=False, authorized_for_submission=False,
        kaggle_acceptance=False, no_end_to_end_speedup_claim=True)

    def persist():
        result['seconds'] = time.monotonic() - started
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        receipt.write_text(text, encoding='utf-8')
        (target / 'RESULT.json').write_text(text, encoding='utf-8')

    persist()
    try:
        for stem in movies:
            locations = []
            for batch in range(8):
                prefix = 'trajectory-event-source-v1-b' + str(batch)
                features = ROOT / '.biohub/cache' / (prefix + '-features')
                frozen = read(features / 'RESULT.json')
                if stem not in frozen['per_movie']:
                    continue
                assert sha(features / 'RESULT.json') == original['source_scope_receipts'][str(batch)]['feature_receipt_sha256']
                base_path = features / (stem + '-prediction.json')
                assert sha(base_path) == frozen['per_movie'][stem]['prediction_sha256']
                backup_path = reports / (prefix + '-full-harvest.json')
                assert sha(backup_path) == original['source_scope_receipts'][str(batch)]['backup_sha256']
                backup = {r['path']: r['sha256'] for r in read(backup_path)['records']}
                origin = ROOT / '.biohub/cache' / (prefix + '-full-output') / (stem + '-original')
                for filename in ('pre-postprocess.json', 'raw-candidates.npz'):
                    assert sha(origin / filename) == backup[stem + '-original/' + filename]
                with np.load(origin / 'raw-candidates.npz', allow_pickle=False) as raw:
                    locations.append((read(origin / 'pre-postprocess.json'), read(base_path), raw['coords'].copy(), raw['edges'].copy()))
            assert len(locations) == 1
            initial, base, coords, edges = locations[0]
            candidate, details = refine(initial, base, coords, edges, model['weights'], per_frame_seconds=10., max_seconds=600.)
            validate_graph(candidate, 100)
            check_event_stage(initial, base, candidate, details, frames=100)
            output = target / (stem + '-prediction.json')
            output.write_text(json.dumps(candidate, sort_keys=True, allow_nan=False), encoding='utf-8')
            row = dict(stem=stem, details=details, prediction_sha256=sha(output))
            result['records'].append(row)
            persist()
            assert not details['solver_fallbacks'] and not details['budget_exhausted']
            previous = ROOT / '.biohub/cache' / reference_name / (stem + '-prediction.json')
            old = proof['inference'][stem]
            assert sha(previous) == old['prediction_sha256'] and candidate == read(previous)
            def comparable(value):
                return {k: ([{a: b for a, b in item.items() if a not in
                    ('allowed_after', 'null_dominated_continuations', 'null_dominated_forks')} for item in v]
                    if k == 'frames' else v) for k, v in value.items() if k != 'seconds'}
            assert comparable(details) == comparable(old['details'])
            row.update(exact_graph_and_decisions=True, reference_seconds=old['details']['seconds'],
                additionally_pruned=sum(r.get('null_dominated_continuations', 0) + r.get('null_dominated_forks', 0) for r in details['frames']))
            persist()
            print(json.dumps({k: v for k, v in row.items() if k != 'details'} | dict(inference_seconds=details['seconds'])), flush=True)
        assert runtime['infer'].__globals__['allowed_options'] is original_pruner
        result['status'] = 'complete_portable_graphs_exact'
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
